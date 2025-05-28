import os
import tarfile

# Root folder containing .tar files and subdirectories
root_folder = r'EQ\0.1g'


def extract_tar_files(root):
    """
    Extracts all .tar files in the root directory.

    Args:
        root (str): Path to the root directory containing .tar files.

    Prints:
        Extraction status for each .tar file.
    """
    for filename in os.listdir(root):
        if filename.endswith('.tar'):
            file_path = os.path.join(root, filename)
            extract_path = os.path.join(root, os.path.splitext(filename)[0])
            print(f'Extracting .tar: {file_path} -> {extract_path}')
            try:
                with tarfile.open(file_path, 'r') as tar:
                    tar.extractall(path=extract_path)
            except Exception as e:
                print(f'Failed to extract {file_path}: {e}')
    print('Finished extracting all .tar files.\n')


def extract_and_remove_targz_files(root):
    """
    Recursively extracts all .tar.gz files under the root directory, then deletes them.

    Args:
        root (str): Path to the root directory.

    Prints:
        Extraction and deletion status for each .tar.gz file.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.tar.gz'):
                file_path = os.path.join(dirpath, filename)
                extract_path = os.path.join(dirpath, os.path.splitext(os.path.splitext(filename)[0])[0])
                print(f'Extracting .tar.gz: {file_path} -> {extract_path}')
                try:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(path=extract_path)
                    os.remove(file_path)
                    print(f'Deleted .tar.gz: {file_path}')
                except Exception as e:
                    print(f'Failed to extract or delete {file_path}: {e}')
    print('Finished extracting and deleting all .tar.gz files.\n')


def delete_rsp_wave_files_in_level2(root):
    """
    Deletes all files ending with .rsp.ps.gz and .wave.ps.gz located in second-level subdirectories of the root.

    Args:
        root (str): Path to the root directory.

    Prints:
        Deletion status for each matching file.
    """
    for level1 in os.listdir(root):
        path1 = os.path.join(root, level1)
        if os.path.isdir(path1):
            for level2 in os.listdir(path1):
                path2 = os.path.join(path1, level2)
                if os.path.isdir(path2):
                    for filename in os.listdir(path2):
                        file_path = os.path.join(path2, filename)
                        if os.path.isfile(file_path) and (
                            filename.endswith('.rsp.ps.gz') or filename.endswith('.wave.ps.gz')
                        ):
                            try:
                                os.remove(file_path)
                                print(f'Deleted: {file_path}')
                            except Exception as e:
                                print(f'Failed to delete {file_path}: {e}')
    print('\nDeleted all .rsp.ps.gz and .wave.ps.gz files in level 2 subdirectories.')


# === MAIN EXECUTION ===
extract_tar_files(root_folder)
extract_and_remove_targz_files(root_folder)
delete_rsp_wave_files_in_level2(root_folder)
