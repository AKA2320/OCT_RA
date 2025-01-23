import config
import inspect

paths = [(name,obj)for name,obj in inspect.getmembers(config) if inspect.isclass(obj)]
# print([name for name,obj in inspect.getmembers(config) if inspect.isclass(obj)])
print(paths[0][1].data_path)