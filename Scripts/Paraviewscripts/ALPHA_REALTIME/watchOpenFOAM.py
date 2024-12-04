import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re

class DirectoryEventHandler(FileSystemEventHandler):
    def __init__(self, case_dir, script_path):
        self.case_dir = case_dir
        self.script_path = script_path

    def on_created(self, event):
        # Only react to directory creation
        if event.is_directory:
            # Extract folder name from the full path
            folder_name = os.path.basename(event.src_path)
            print(f"New folder created: {folder_name}")

            # Check if the created folder is a valid OpenFOAM time directory
            if re.match(r'^\d+(\.\d+)?$', folder_name):  # Time directories like 0, 1, 0.1
                print(f"New time directory detected: {folder_name}")
                # Run pvpython with the folder name as desired_time
                self.run_pvpython(folder_name)

    def run_pvpython(self, folder_name):
        # Run the pvpython script with the folder name as argument for desired_time
        print(f"Running pvpython with desired time: {folder_name}")
        subprocess.run(['pvpython', self.script_path, '--desired_time', folder_name, '--case', self.case_dir])


def monitor_case(case_dir, script_path):
    # Set up the event handler and observer
    event_handler = DirectoryEventHandler(case_dir, script_path)
    observer = Observer()
    observer.schedule(event_handler, path=case_dir, recursive=True)  # Watch subdirectories recursively
    
    # Start monitoring
    observer.start()
    print(f"Watching for new directories or files in {case_dir}...")
    
    try:
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Set the OpenFOAM case directory and pvpython script path
    case_dir = os.getcwd()  # Current directory (OpenFOAM case)
    script_path = 'REALTIME_Calculate_AvVel_multi_multicase.py'  # Provide your pvpython script path

    monitor_case(case_dir, script_path)
