import asyncio
from typing import Dict, List, Callable
from datetime import datetime
class FallbackSystem:
    def __init__(self):
        self.primary_systems: Dict[str, Dict] = {}
        self.backup_systems: Dict[str, List[Dict]] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.active_system: Dict[str, str] = {}
    def register_primary(self, system_name: str, system: Dict):
        self.primary_systems[system_name] = system
        self.active_system[system_name] = 'primary'
    def register_backup(self, system_name: str, backup: Dict):
        if system_name not in self.backup_systems:
            self.backup_systems[system_name] = []
        self.backup_systems[system_name].append(backup)
    def register_health_check(self, system_name: str, check_func: Callable):
        self.health_checks[system_name] = check_func
    async def monitor_systems(self):
        while True:
            for system_name in self.primary_systems:
                try:
                    health_ok = await self._check_health(system_name)
                    if not health_ok:
                        await self._activate_backup(system_name)
                except Exception as e:
                    print(f"Erreur monitoring {system_name}: {e}")
                    await self._activate_backup(system_name)
            await asyncio.sleep(5)  # Check every 5 seconds
    async def _check_health(self, system_name: str) -> bool:
        if system_name not in self.health_checks:
            return True
        try:
            return await self.health_checks[system_name]()
        except:
            return False
    async def _activate_backup(self, system_name: str):
        if system_name not in self.backup_systems:
            print(f"No backup available for {system_name}")
            return
        current_index = 0
        for backup in self.backup_systems[system_name]:
            try:
                # Tentative d'activation du backup
                if await self._test_backup(backup):
                    self.active_system[system_name] = f'backup_{current_index}'
                    print(f"Activated backup_{current_index} for {system_name}")
                    return
            except Exception as e:
                print(f"Backup {current_index} failed: {e}")
            current_index += 1
    async def _test_backup(self, backup: Dict) -> bool:
        # Test de connexion/fonctionnement du backup
        try:
            if 'test_func' in backup:
                return await backup['test_func']()
            return True
        except:
            return False
