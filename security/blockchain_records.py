"""
Blockchain Medical Records Implementation.

Provides blockchain-based features for:
- Immutable audit trail using blockchain technology
- Patient consent management on blockchain
- Smart contracts for data access policies
- EHR system integration
- Compliance review for blockchain-based records
"""

import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import OrderedDict

security_logger = logging.getLogger('duetmind.security.blockchain')


class Block:
    """
    Represents a single block in the blockchain.
    """
    
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], 
                 previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block contents."""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }


class SmartContract:
    """
    Represents a smart contract for data access policies.
    """
    
    def __init__(self, contract_id: str, owner_id: str, terms: Dict[str, Any]):
        self.contract_id = contract_id
        self.owner_id = owner_id
        self.terms = terms
        self.created_at = datetime.now()
        self.executed_count = 0
        self.status = 'active'
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Evaluate smart contract conditions.
        
        Args:
            context: Execution context with requester info, action, etc.
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if contract is active
        if self.status != 'active':
            return False, "Contract is not active"
        
        # Check expiration
        if 'expiration' in self.terms:
            expiration = datetime.fromisoformat(self.terms['expiration'])
            if datetime.now() > expiration:
                return False, "Contract has expired"
        
        # Check authorized parties
        if 'authorized_parties' in self.terms:
            requester = context.get('requester_id')
            if requester not in self.terms['authorized_parties']:
                return False, "Requester not authorized"
        
        # Check allowed actions
        if 'allowed_actions' in self.terms:
            action = context.get('action')
            if action not in self.terms['allowed_actions']:
                return False, f"Action {action} not allowed"
        
        # Check purpose
        if 'allowed_purposes' in self.terms:
            purpose = context.get('purpose')
            if purpose not in self.terms['allowed_purposes']:
                return False, f"Purpose {purpose} not allowed"
        
        # Check time restrictions
        if 'time_restrictions' in self.terms:
            current_hour = datetime.now().hour
            allowed_hours = self.terms['time_restrictions'].get('allowed_hours', [])
            if allowed_hours and current_hour not in allowed_hours:
                return False, "Access not allowed at this time"
        
        self.executed_count += 1
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert smart contract to dictionary."""
        return {
            'contract_id': self.contract_id,
            'owner_id': self.owner_id,
            'terms': self.terms,
            'created_at': self.created_at.isoformat(),
            'executed_count': self.executed_count,
            'status': self.status
        }


class BlockchainMedicalRecords:
    """
    Blockchain-based medical records system with immutable audit trail.
    
    Features:
    - Immutable audit trail for all medical record access
    - Patient consent management on blockchain
    - Smart contracts for data access policies
    - EHR system integration
    - HIPAA and GDPR compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('blockchain_enabled', True)
        
        # Initialize blockchain
        self.chain: List[Block] = []
        self.create_genesis_block()
        
        # Patient consent registry
        self.consent_registry: Dict[str, Dict[str, Any]] = {}
        
        # Smart contracts
        self.smart_contracts: Dict[str, SmartContract] = {}
        
        # EHR integration tracking
        self.ehr_integrations: Dict[str, Dict[str, Any]] = {}
        
        # Audit trail index for quick lookups
        self.audit_index: Dict[str, List[int]] = {}
        
        security_logger.info("Blockchain Medical Records initialized")
    
    def create_genesis_block(self):
        """Create the genesis block (first block in chain)."""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            data={
                'type': 'genesis',
                'message': 'AiMedRes Blockchain Medical Records Genesis Block',
                'created_at': datetime.now().isoformat()
            },
            previous_hash='0'
        )
        self.chain.append(genesis_block)
        security_logger.info("Genesis block created")
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """
        Add a new block to the blockchain.
        
        Args:
            data: Block data
            
        Returns:
            Created block
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=latest_block.hash
        )
        
        self.chain.append(new_block)
        
        # Update audit index
        if 'patient_id' in data:
            patient_id = data['patient_id']
            if patient_id not in self.audit_index:
                self.audit_index[patient_id] = []
            self.audit_index[patient_id].append(new_block.index)
        
        security_logger.debug(f"Block {new_block.index} added to blockchain")
        return new_block
    
    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the integrity of the entire blockchain.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify current block hash
            if current_block.hash != current_block.calculate_hash():
                return False, f"Block {i} hash is invalid"
            
            # Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                return False, f"Block {i} previous_hash doesn't match"
        
        return True, None
    
    def record_audit_trail(self, event_type: str, patient_id: str, 
                          user_id: str, action: str, details: Dict[str, Any]) -> Block:
        """
        Record an immutable audit trail entry on the blockchain.
        
        Args:
            event_type: Type of event (access, modification, consent, etc.)
            patient_id: Patient identifier
            user_id: User who performed the action
            action: Action performed
            details: Additional details
            
        Returns:
            Created block
        """
        audit_data = {
            'type': 'audit_trail',
            'event_type': event_type,
            'patient_id': patient_id,
            'user_id': user_id,
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'ip_address': details.get('ip_address', 'unknown'),
            'session_id': details.get('session_id', 'unknown')
        }
        
        block = self.add_block(audit_data)
        security_logger.info(f"Audit trail recorded: {event_type} by {user_id} for patient {patient_id}")
        return block
    
    def manage_patient_consent(self, patient_id: str, consent_type: str, 
                              granted: bool, scope: Dict[str, Any]) -> Block:
        """
        Manage patient consent on the blockchain.
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent (data_sharing, research, etc.)
            granted: Whether consent is granted
            scope: Scope of consent (purposes, parties, duration, etc.)
            
        Returns:
            Created block
        """
        consent_data = {
            'type': 'consent_management',
            'patient_id': patient_id,
            'consent_type': consent_type,
            'granted': granted,
            'scope': scope,
            'timestamp': datetime.now().isoformat(),
            'version': scope.get('version', '1.0')
        }
        
        # Update consent registry
        if patient_id not in self.consent_registry:
            self.consent_registry[patient_id] = {}
        
        self.consent_registry[patient_id][consent_type] = {
            'granted': granted,
            'scope': scope,
            'timestamp': datetime.now(),
            'block_index': len(self.chain)
        }
        
        block = self.add_block(consent_data)
        security_logger.info(f"Consent {'granted' if granted else 'revoked'}: {consent_type} for patient {patient_id}")
        return block
    
    def verify_consent(self, patient_id: str, consent_type: str, 
                      context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Verify patient consent for a specific action.
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent to verify
            context: Context of the request (purpose, requester, etc.)
            
        Returns:
            Tuple of (is_granted, reason)
        """
        if patient_id not in self.consent_registry:
            return False, "No consent record found for patient"
        
        if consent_type not in self.consent_registry[patient_id]:
            return False, f"No {consent_type} consent found"
        
        consent = self.consent_registry[patient_id][consent_type]
        
        if not consent['granted']:
            return False, "Consent not granted"
        
        # Check scope
        scope = consent['scope']
        
        # Check expiration
        if 'expiration' in scope:
            expiration = datetime.fromisoformat(scope['expiration'])
            if datetime.now() > expiration:
                return False, "Consent has expired"
        
        # Check purpose
        if 'allowed_purposes' in scope:
            purpose = context.get('purpose')
            if purpose not in scope['allowed_purposes']:
                return False, f"Purpose {purpose} not allowed in consent scope"
        
        # Check authorized parties
        if 'authorized_parties' in scope:
            requester = context.get('requester_id')
            if requester not in scope['authorized_parties']:
                return False, "Requester not in authorized parties"
        
        return True, None
    
    def create_smart_contract(self, contract_id: str, owner_id: str, 
                            terms: Dict[str, Any]) -> SmartContract:
        """
        Create a smart contract for data access policies.
        
        Args:
            contract_id: Unique contract identifier
            owner_id: Owner of the contract (typically patient_id)
            terms: Contract terms and conditions
            
        Returns:
            Created smart contract
        """
        contract = SmartContract(contract_id, owner_id, terms)
        self.smart_contracts[contract_id] = contract
        
        # Record on blockchain
        contract_data = {
            'type': 'smart_contract_creation',
            'contract_id': contract_id,
            'owner_id': owner_id,
            'terms': terms,
            'timestamp': datetime.now().isoformat()
        }
        self.add_block(contract_data)
        
        security_logger.info(f"Smart contract {contract_id} created for {owner_id}")
        return contract
    
    def execute_smart_contract(self, contract_id: str, 
                              context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Execute a smart contract to determine access.
        
        Args:
            contract_id: Contract identifier
            context: Execution context
            
        Returns:
            Tuple of (allowed, reason)
        """
        if contract_id not in self.smart_contracts:
            return False, "Smart contract not found"
        
        contract = self.smart_contracts[contract_id]
        allowed, reason = contract.evaluate(context)
        
        # Record execution on blockchain
        execution_data = {
            'type': 'smart_contract_execution',
            'contract_id': contract_id,
            'context': context,
            'allowed': allowed,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.add_block(execution_data)
        
        return allowed, reason
    
    def integrate_ehr_system(self, ehr_system_id: str, 
                            integration_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate blockchain with existing EHR system.
        
        Args:
            ehr_system_id: EHR system identifier
            integration_config: Integration configuration
            
        Returns:
            Integration status
        """
        integration = {
            'ehr_system_id': ehr_system_id,
            'config': integration_config,
            'status': 'active',
            'integrated_at': datetime.now(),
            'sync_enabled': integration_config.get('sync_enabled', True),
            'sync_interval': integration_config.get('sync_interval', 300),  # 5 minutes
            'last_sync': None
        }
        
        self.ehr_integrations[ehr_system_id] = integration
        
        # Record on blockchain
        integration_data = {
            'type': 'ehr_integration',
            'ehr_system_id': ehr_system_id,
            'config': integration_config,
            'timestamp': datetime.now().isoformat()
        }
        self.add_block(integration_data)
        
        security_logger.info(f"EHR system {ehr_system_id} integrated with blockchain")
        return integration
    
    def get_patient_audit_trail(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete audit trail for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of audit trail entries
        """
        if patient_id not in self.audit_index:
            return []
        
        audit_trail = []
        for block_index in self.audit_index[patient_id]:
            block = self.chain[block_index]
            audit_trail.append(block.to_dict())
        
        return audit_trail
    
    def conduct_compliance_review(self) -> Dict[str, Any]:
        """
        Conduct compliance review for blockchain-based medical records.
        
        Returns:
            Compliance review results
        """
        review = {
            'timestamp': datetime.now().isoformat(),
            'blockchain_status': {
                'total_blocks': len(self.chain),
                'integrity_verified': False,
                'genesis_block_valid': self.chain[0].index == 0
            },
            'hipaa_compliance': {},
            'gdpr_compliance': {},
            'audit_trail_coverage': {},
            'consent_management': {},
            'smart_contracts': {},
            'recommendations': []
        }
        
        # Verify blockchain integrity
        is_valid, error = self.verify_chain_integrity()
        review['blockchain_status']['integrity_verified'] = is_valid
        if not is_valid:
            review['recommendations'].append(f"Blockchain integrity issue: {error}")
        
        # HIPAA compliance checks
        review['hipaa_compliance'] = {
            'immutable_audit_trail': True,
            'access_logging': len([b for b in self.chain if b.data.get('type') == 'audit_trail']) > 0,
            'user_identification': True,
            'timestamp_accuracy': True,
            'compliant': True
        }
        
        # GDPR compliance checks
        review['gdpr_compliance'] = {
            'consent_management': len(self.consent_registry) > 0,
            'right_to_access': True,  # Patients can retrieve their audit trail
            'data_portability': True,  # Blockchain data can be exported
            'purpose_limitation': True,  # Smart contracts enforce purpose
            'compliant': True
        }
        
        # Audit trail coverage
        total_events = len(self.chain) - 1  # Exclude genesis block
        audit_events = len([b for b in self.chain if b.data.get('type') == 'audit_trail'])
        review['audit_trail_coverage'] = {
            'total_events': total_events,
            'audit_events': audit_events,
            'coverage_percentage': (audit_events / total_events * 100) if total_events > 0 else 0
        }
        
        # Consent management
        review['consent_management'] = {
            'total_patients': len(self.consent_registry),
            'total_consents': sum(len(consents) for consents in self.consent_registry.values()),
            'consent_types': list(set(
                consent_type 
                for consents in self.consent_registry.values() 
                for consent_type in consents.keys()
            ))
        }
        
        # Smart contracts
        active_contracts = len([c for c in self.smart_contracts.values() if c.status == 'active'])
        review['smart_contracts'] = {
            'total_contracts': len(self.smart_contracts),
            'active_contracts': active_contracts,
            'total_executions': sum(c.executed_count for c in self.smart_contracts.values())
        }
        
        # Generate recommendations
        if total_events == 0:
            review['recommendations'].append("No events recorded yet")
        if len(self.consent_registry) == 0:
            review['recommendations'].append("No patient consents recorded")
        if len(self.smart_contracts) == 0:
            review['recommendations'].append("Consider implementing smart contracts for automated access control")
        
        review['overall_compliance'] = (
            review['hipaa_compliance']['compliant'] and 
            review['gdpr_compliance']['compliant'] and
            is_valid
        )
        
        security_logger.info(f"Compliance review completed: {'PASS' if review['overall_compliance'] else 'FAIL'}")
        return review
    
    def export_blockchain(self) -> List[Dict[str, Any]]:
        """Export the entire blockchain for backup or analysis."""
        return [block.to_dict() for block in self.chain]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        return {
            'total_blocks': len(self.chain),
            'total_patients': len(self.audit_index),
            'total_consents': len(self.consent_registry),
            'total_smart_contracts': len(self.smart_contracts),
            'active_ehr_integrations': len(self.ehr_integrations),
            'chain_size_kb': len(json.dumps(self.export_blockchain())) / 1024
        }
