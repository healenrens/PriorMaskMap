import pickle
import threading
import os
import hashlib
import random


class MapModificationDatabase:
    def __init__(
        self,
        use_cache=False,
        cache_dir=".cache_data",
        max_cache_size=1000,
        random_seed=None,
    ):
        self._data = {}
        self._lock = threading.Lock()
        self._loaded = False
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self._cache_index = {}
        self._lru_cache = {}
        self._cache_access_order = []
        self.random_state = random.Random(random_seed)
        self.random_seed = random_seed

    def _get_cache_key(self, data_path_list):
        if isinstance(data_path_list, str):
            data_path_list = [data_path_list]
        path_info = []
        for path in data_path_list:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                path_info.append(f"{path}:{mtime}")
            else:
                path_info.append(path)
        combined_str = "|".join(sorted(path_info))
        return hashlib.md5(combined_str.encode()).hexdigest()

    def _generate_cache_filename(self, frame_token, mod_type, change_class, params):
        safe_frame = str(frame_token).replace("/", "_").replace("\\", "_")
        safe_mod = str(mod_type).replace("/", "_").replace("\\", "_")
        safe_class = str(change_class).replace("/", "_").replace("\\", "_")
        safe_params = str(params).replace("/", "_").replace("\\", "_")
        filename = f"{safe_frame}_{safe_mod}_{safe_class}_{safe_params}.pkl"
        if len(filename) > 200:
            content = f"{frame_token}_{mod_type}_{change_class}_{params}"
            filename = hashlib.md5(content.encode()).hexdigest() + ".pkl"
        return filename

    def _extract_full_keys(self, data_dict, prefix_path=""):
        full_keys = {}
        for frame_token, frame_data in data_dict.items():
            if not isinstance(frame_data, dict):
                continue
            for mod_type, mod_data in frame_data.items():
                if not isinstance(mod_data, dict):
                    if "_" in mod_type:
                        parts = mod_type.split("_", 2)
                        if len(parts) >= 3:
                            actual_mod_type, change_class, params = parts
                            key = (frame_token, actual_mod_type, change_class, params)
                            full_keys[key] = mod_data
                    continue
                for change_class, class_data in mod_data.items():
                    if not isinstance(class_data, dict):
                        key = (frame_token, mod_type, change_class, "default")
                        full_keys[key] = class_data
                        continue
                    for params, param_data in class_data.items():
                        key = (frame_token, mod_type, change_class, params)
                        full_keys[key] = param_data
        return full_keys

    def _reconstruct_nested_dict(self, full_keys_dict):
        result = {}
        for (
            frame_token,
            mod_type,
            change_class,
            params,
        ), data in full_keys_dict.items():
            if frame_token not in result:
                result[frame_token] = {}
            if mod_type not in result[frame_token]:
                result[frame_token][mod_type] = {}
            if change_class not in result[frame_token][mod_type]:
                result[frame_token][mod_type][change_class] = {}
            result[frame_token][mod_type][change_class][params] = data
        return result

    def _smart_merge_data(self, data_list):
        if not data_list:
            return {}
        if len(data_list) == 1:
            return self._extract_full_keys(data_list[0])
        all_full_keys = {}
        conflict_keys = {}
        for i, data_dict in enumerate(data_list):
            current_full_keys = self._extract_full_keys(data_dict)
            for key, value in current_full_keys.items():
                if key in all_full_keys:
                    if key not in conflict_keys:
                        conflict_keys[key] = [all_full_keys[key]]
                    conflict_keys[key].append(value)
                else:
                    all_full_keys[key] = value
        self._conflict_info = {
            "total_conflicts": len(conflict_keys),
            "conflict_keys": list(conflict_keys.keys()),
            "random_seed": self.random_seed,
        }
        for key, candidates in conflict_keys.items():
            selected = self.random_state.choice(candidates)
            all_full_keys[key] = selected
        return all_full_keys

    def get_conflict_info(self):
        return getattr(
            self,
            "_conflict_info",
            {
                "total_conflicts": 0,
                "conflict_keys": [],
                "random_seed": self.random_seed,
            },
        )

    def _create_cache_from_pkl(self, data_path_list, cache_key):
        cache_subdir = os.path.join(self.cache_dir, cache_key)
        os.makedirs(cache_subdir, exist_ok=True)
        index_file = os.path.join(cache_subdir, "index.pkl")
        if os.path.exists(index_file):
            with open(index_file, "rb") as f:
                self._cache_index = pickle.load(f)
            return
        data_list = []
        for data_path in data_path_list:
            with open(data_path, "rb") as f:
                loaded_chunk = pickle.load(f)
                data_list.append(loaded_chunk)
        merged_full_keys = self._smart_merge_data(data_list)
        for (
            frame_token,
            mod_type,
            change_class,
            params,
        ), data in merged_full_keys.items():
            filename = self._generate_cache_filename(
                frame_token, mod_type, change_class, params
            )
            cache_file = os.path.join(cache_subdir, filename)
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            key_tuple = (frame_token, mod_type, change_class, params)
            self._cache_index[key_tuple] = cache_file
        with open(index_file, "wb") as f:
            pickle.dump(self._cache_index, f)

    def _load_specific_data(self, frame_token, mod_type, change_class, params):
        key_tuple = (frame_token, mod_type, change_class, params)
        if key_tuple in self._lru_cache:
            self._cache_access_order.remove(key_tuple)
            self._cache_access_order.append(key_tuple)
            return self._lru_cache[key_tuple]
        cache_file = self._cache_index[key_tuple]
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        self._lru_cache[key_tuple] = data
        self._cache_access_order.append(key_tuple)
        while len(self._lru_cache) > self.max_cache_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._lru_cache[oldest_key]
        return data

    def _deep_merge_dict(self, target_dict: dict, source_dict: dict) -> None:
        for key, value in source_dict.items():
            if key in target_dict:
                if isinstance(target_dict[key], dict) and isinstance(value, dict):
                    self._deep_merge_dict(target_dict[key], value)
                else:
                    target_dict[key] = value
            else:
                target_dict[key] = value

    def load_data(self, data_path):
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                path_list = data_path if isinstance(data_path, list) else [data_path]
                if self.use_cache:
                    cache_key = self._get_cache_key(path_list)
                    self._create_cache_from_pkl(path_list, cache_key)
                else:
                    data_list = []
                    for data_path in path_list:
                        with open(data_path, "rb") as f:
                            loaded_chunk = pickle.load(f)
                            data_list.append(loaded_chunk)
                    self._data = self._smart_merge_data(data_list)
                self._loaded = True

    def get_modification(self, frame_token, mod_type, change_class, params):
        if self.use_cache:
            return self._load_specific_data(frame_token, mod_type, change_class, params)
        else:
            key_tuple = (frame_token, mod_type, change_class, params)
            return self._data[key_tuple]


class Modmap:
    def __init__(
        self,
        data_path,
        skip=False,
        use_cache=True,
        cache_dir=".cache_data",
        max_cache_size=1000,
        random_seed=None,
    ):
        self.mapdata = MapModificationDatabase(
            use_cache=use_cache,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            random_seed=random_seed,
        )
        self.mapdata.load_data(data_path)
        self.skip = skip

    def get(self, frame, config, return_vec=False):
        entry = self.mapdata.get_modification(frame, config[0], config[1], config[2])
        if return_vec:
            return (
                entry["rasterized_map"],
                entry["change_rate"],
                entry["original_map"],
            )
        return (entry["rasterized_map"], entry["change_rate"])
