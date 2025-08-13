import git
import os
import json
from datetime import datetime
import shutil
class GitAutoSaver:
    def __init__(self, repo_path='.'):
        self.repo = git.Repo(repo_path)
        self.backup_dir = os.path.join(repo_path, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
    def create_snapshot(self, message=None):
        message = message or f"Auto-save {timestamp}"
        self.repo.git.add(all=True)
        try:
            self.repo.index.commit(message)
            self.repo.create_tag(f"v{timestamp}", message=message)
            self._create_backup_archive(f"v{timestamp}")
            return True
        except git.exc.GitError:
            return False
    def _create_backup_archive(self, version_tag):
        backup_file = os.path.join(self.backup_dir, f"backup_{version_tag}.zip")
        with open('git_files.txt', 'w') as f:
            for item in self.repo.git.ls_files().split('\n'):
                if item: f.write(f"{item}\n")
        shutil.make_archive(
            backup_file.replace('.zip', ''),
            'zip',
            root_dir=self.repo.working_dir
        )
        os.remove('git_files.txt')
