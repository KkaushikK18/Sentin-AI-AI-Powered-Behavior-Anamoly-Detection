import pandas as pd
import numpy as np
from faker import Faker
import random
import yaml
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BehaviorDataGenerator:
    """
    Generates synthetic behavioral logs for cybersecurity anomaly detection.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['data_generation']
        
        self.fake = Faker()
        Faker.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        random.seed(self.config.get('random_seed', 42))
        
        self.entity_types = ['User', 'Admin', 'Service Account', 'IoT Device', 
                             'Industrial Edge Device', 'Server', 'Laptop', 'Mobile Device']
        
        self.protocols = ['HTTP', 'HTTPS', 'SSH', 'RDP', 'SMB', 'FTP', 'SQL']
        self.auth_methods = ['Password', 'MFA', 'OAuth', 'Kerberos', 'Key', 'Biometric']
        self.os_list = ['Windows 10', 'Windows 11', 'macOS', 'Ubuntu', 'CentOS', 'Android', 'iOS', 'Embedded Linux']
        self.browsers = ['Chrome', 'Firefox', 'Edge', 'Safari', 'None']
        
        self.resources = [f"/api/v1/resource_{i}" for i in range(20)] + \
                         [f"db_table_{i}" for i in range(10)] + \
                         [f"\\\\server\\share_{i}" for i in range(5)]
        
        self.entities = self._generate_entities()
        self.logs = []

    def _generate_entities(self) -> List[Dict[str, Any]]:
        """Generate base profiles for entities to establish 'normal' behavior."""
        entities = []
        num_total = self.config['num_users'] + self.config['num_service_accounts'] + self.config['num_devices']
        
        for i in range(num_total):
            if i < self.config['num_users']:
                e_type = np.random.choice(['User', 'Admin'], p=[0.9, 0.1])
            elif i < self.config['num_users'] + self.config['num_service_accounts']:
                e_type = 'Service Account'
            else:
                e_type = np.random.choice(['IoT Device', 'Industrial Edge Device', 'Server', 'Laptop', 'Mobile Device'])
            
            # Normal profile attributes
            profile = {
                'entity_id': f"ENT_{i:05d}",
                'entity_type': e_type,
                'home_country': self.fake.country(),
                'home_city': self.fake.city(),
                'primary_ip': self.fake.ipv4(),
                'working_hours': (np.random.randint(6, 10), np.random.randint(15, 20)), # start, end
                'typical_os': np.random.choice(self.os_list),
                'typical_browser': np.random.choice(self.browsers) if e_type in ['User', 'Admin'] else 'None',
                'typical_auth': np.random.choice(self.auth_methods),
                'typical_resources': list(np.random.choice(self.resources, size=np.random.randint(2, 6), replace=False))
            }
            entities.append(profile)
        return entities

    def _generate_normal_events(self, entity: Dict[str, Any], current_time: datetime) -> Dict[str, Any]:
        """Generates a single normal event for an entity."""
        # Check if active during working hours (mostly)
        hour = current_time.hour
        is_working_hour = entity['working_hours'][0] <= hour <= entity['working_hours'][1]
        
        # Occasional off-hours activity for normal users, highly active for servers
        if not is_working_hour and entity['entity_type'] in ['User', 'Admin'] and random.random() > 0.1:
            return None # Skip event

        success = np.random.choice([1, 0], p=[0.95, 0.05]) # Mostly successful logins
        
        event = {
            'entity_id': entity['entity_id'],
            'entity_type': entity['entity_type'],
            'timestamp': current_time,
            'source_ip': entity['primary_ip'] if random.random() > 0.1 else self.fake.ipv4(),
            'country': entity['home_country'] if random.random() > 0.05 else self.fake.country(),
            'city': entity['home_city'] if random.random() > 0.05 else self.fake.city(),
            'resource_accessed': np.random.choice(entity['typical_resources']) if random.random() > 0.1 else np.random.choice(self.resources),
            'auth_method': entity['typical_auth'] if random.random() > 0.05 else np.random.choice(self.auth_methods),
            'session_duration': np.random.exponential(scale=300) if success else 0, # seconds
            'command_sequence': f"CMD_{np.random.randint(1, 100)}",
            'device_fingerprint': f"{entity['typical_os']}_{entity['typical_browser']}",
            'login_success': success,
            'protocol': np.random.choice(self.protocols),
            'operating_system': entity['typical_os'],
            'browser': entity['typical_browser'],
            'device_type': entity['entity_type'] if 'Device' in entity['entity_type'] else 'Workstation',
            'label': 0, # Normal
            'attack_type': 'None'
        }
        
        # Derived geo location for simplicity
        event['geo_location'] = f"{event['country']}, {event['city']}"
        event['hour_of_day'] = current_time.hour
        event['day_of_week'] = current_time.weekday()
        
        # Placeholder for complex risk features (to be computed in Feature Engineering)
        event['risk_features'] = "{}" 
        
        return event

    def generate_dataset(self):
        """Generates the full dataset with normal events and injected anomalies."""
        logging.info("Starting data generation...")
        start_date = datetime.now() - timedelta(days=self.config['num_days'])
        
        for entity in self.entities:
            # Generate normal background traffic
            num_events = int(np.random.normal(self.config['events_per_day_per_user_mean'], 
                                              self.config['events_per_day_per_user_std']) * self.config['num_days'])
            num_events = max(10, num_events) # ensure some events
            
            timestamps = [start_date + timedelta(seconds=random.randint(0, self.config['num_days'] * 86400)) for _ in range(num_events)]
            timestamps.sort()
            
            for ts in timestamps:
                event = self._generate_normal_events(entity, ts)
                if event:
                    self.logs.append(event)
                    
        logging.info(f"Generated {len(self.logs)} normal events. Injecting anomalies...")
        self._inject_anomalies()
        
        df = pd.DataFrame(self.logs)
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save to file
        df.to_csv(self.config['out_file'], index=False)
        logging.info(f"Dataset generated and saved to {self.config['out_file']} with {len(df)} records.")
        return df

    def _inject_anomalies(self):
        """Inject specific cyber attack patterns into the logs."""
        num_normal = len(self.logs)
        probs = self.config['attack_probabilities']
        
        # 1. Brute Force
        bf_count = int(num_normal * probs.get('Brute Force', 0.0))
        for _ in range(bf_count // 10): # group by 10 attempts
            target_entity = random.choice(self.entities)
            ts = datetime.now() - timedelta(days=random.randint(0, self.config['num_days']))
            attacker_ip = self.fake.ipv4()
            for _ in range(10):
                ts += timedelta(seconds=random.randint(1, 5))
                self.logs.append(self._create_anomaly_event(target_entity, ts, 'Brute Force', login_success=0, source_ip=attacker_ip))

        # 2. Impossible Travel
        it_count = int(num_normal * probs.get('Impossible Travel', 0.0))
        for _ in range(it_count // 2):
            target_entity = random.choice([e for e in self.entities if e['entity_type'] in ['User', 'Admin']])
            ts1 = datetime.now() - timedelta(days=random.randint(0, self.config['num_days']))
            ts2 = ts1 + timedelta(minutes=random.randint(5, 30)) # impossible short time
            self.logs.append(self._create_anomaly_event(target_entity, ts1, 'Impossible Travel', country='USA', city='New York'))
            self.logs.append(self._create_anomaly_event(target_entity, ts2, 'Impossible Travel', country='China', city='Beijing'))

        # 3. Lateral Movement
        lm_count = int(num_normal * probs.get('Lateral Movement', 0.0))
        for _ in range(lm_count):
            target_entity = random.choice(self.entities)
            ts = datetime.now() - timedelta(days=random.randint(0, self.config['num_days']))
            # Accessing non-typical resource
            novel_resource = random.choice([r for r in self.resources if r not in target_entity['typical_resources']])
            self.logs.append(self._create_anomaly_event(target_entity, ts, 'Lateral Movement', resource_accessed=novel_resource, login_success=1))

        # 4. Device Spoofing
        ds_count = int(num_normal * probs.get('Device Spoofing', 0.0))
        for _ in range(ds_count):
            target_entity = random.choice(self.entities)
            ts = datetime.now() - timedelta(days=random.randint(0, self.config['num_days']))
            new_os = random.choice([os for os in self.os_list if os != target_entity['typical_os']])
            self.logs.append(self._create_anomaly_event(target_entity, ts, 'Device Spoofing', operating_system=new_os, device_fingerprint=f"{new_os}_Spoof"))

        # Add other attacks similarly (Credential Stuffing)
        cs_count = int(num_normal * probs.get('Credential Stuffing', 0.0))
        for _ in range(cs_count):
            target_entity = random.choice(self.entities)
            ts = datetime.now() - timedelta(days=random.randint(0, self.config['num_days']))
            self.logs.append(self._create_anomaly_event(target_entity, ts, 'Credential Stuffing', login_success=0, auth_method='Password'))
            
        # 6. Low and Slow Data Exfiltration
        lse_count = int(num_normal * probs.get('Low-and-Slow Exfiltration', 0.0))
        for _ in range(max(1, lse_count // 5)): # Sequence of 5 events across days
            target_entity = random.choice([e for e in self.entities if e['entity_type'] in ['User', 'Admin']])
            ts = datetime.now() - timedelta(days=random.randint(5, self.config['num_days']))
            for _ in range(5):
                ts += timedelta(days=random.randint(1, 3), hours=random.randint(0, 4)) # Off-hours access
                self.logs.append(self._create_anomaly_event(target_entity, ts, 'Low-and-Slow Exfiltration', resource_accessed="sensitive_db", session_duration=random.randint(10, 60)))
                
        # 7. Insider Drift
        id_count = int(num_normal * probs.get('Insider Drift', 0.0))
        for _ in range(max(1, id_count // 5)): # Gradual privilege expansion
            target_entity = random.choice([e for e in self.entities if e['entity_type'] == 'User'])
            ts = datetime.now() - timedelta(days=random.randint(5, self.config['num_days']))
            for _ in range(5):
                ts += timedelta(days=random.randint(1, 3))
                novel_resource = random.choice([r for r in self.resources if r not in target_entity['typical_resources']])
                self.logs.append(self._create_anomaly_event(target_entity, ts, 'Insider Drift', resource_accessed=novel_resource, command_sequence="CMD_ELEVATE"))

    def _create_anomaly_event(self, entity, ts, attack_type, **kwargs):
        """Helper to create an anomaly event overriding defaults."""
        event = self._generate_normal_events(entity, ts)
        if event is None: # if it was filtered out by off-hours logic
             event = {
                'entity_id': entity['entity_id'], 'entity_type': entity['entity_type'], 'timestamp': ts,
                'source_ip': entity['primary_ip'], 'country': entity['home_country'], 'city': entity['home_city'],
                'resource_accessed': entity['typical_resources'][0], 'auth_method': entity['typical_auth'],
                'session_duration': 100, 'command_sequence': "CMD_X", 'device_fingerprint': f"{entity['typical_os']}_X",
                'login_success': 1, 'protocol': 'HTTPS', 'operating_system': entity['typical_os'],
                'browser': entity['typical_browser'], 'device_type': 'Workstation',
             }
        
        event['label'] = 1
        event['attack_type'] = attack_type
        
        # Apply overrides
        for k, v in kwargs.items():
            event[k] = v
            
        event['geo_location'] = f"{event.get('country', entity['home_country'])}, {event.get('city', entity['home_city'])}"
        event['hour_of_day'] = ts.hour
        event['day_of_week'] = ts.weekday()
        
        return event

if __name__ == "__main__":
    generator = BehaviorDataGenerator()
    generator.generate_dataset()
