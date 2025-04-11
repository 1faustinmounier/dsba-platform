import os
import sys
import inspect
import importlib.util

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def generate_architecture_documentation(project_path):
    documentation = ""
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                documentation += f"\nFile: {file_path}\n"
                module_name = file[:-3]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    error_message = str(e)
                    if "attempted relative import" in error_message:
                        documentation += f"  [{file} file has been ignored because it uses relative imports with no known parent.]\n"
                    else:
                        documentation += f"  [{file} file was ignored because an error occurred during loading.]\n"
                    continue
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    try:
                        source = inspect.getsourcefile(obj)
                    except TypeError:
                        source = None
                    if source and os.path.abspath(source) == os.path.abspath(file_path):
                        documentation += f"  Class: {name}\n"
                        for meth_name, meth_obj in inspect.getmembers(obj, inspect.isfunction):
                            documentation += f"    Method: {meth_name}\n"
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    try:
                        source = inspect.getsourcefile(obj)
                    except TypeError:
                        source = None
                    if source and os.path.abspath(source) == os.path.abspath(file_path):
                        documentation += f"  Function: {name}\n"
    output_path = os.path.join(project_root, "documentation_architecture.txt")
    with open(output_path, "w", encoding="utf-8") as doc_file:
        doc_file.write(documentation)
    print(f"Documentation generated in : {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.join(current_dir, "..", "src")
    generate_architecture_documentation(os.path.abspath(project_dir))