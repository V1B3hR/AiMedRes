"""Script to generate realistic, de-identified synthetic benchmark fixtures for open-access datasets (OASIS-2, PRO-ACT ALS, PTB-XL ECG)."""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_oasis_fixture(output_path: Path, n_subjects: int = 80, random_seed: int = 42):
    np.random.seed(random_seed)
    records = []
    
    for i in range(1, n_subjects + 1):
        subj_id = f"OAS2_{i:04d}"
        gender = np.random.choice(["M", "F"], p=[0.42, 0.58])
        hand = "R"
        educ = int(np.clip(np.random.normal(14.5, 3.0), 6, 23))
        ses = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.20, 0.15]), 1, 5))
        base_age = np.clip(np.random.normal(75.0, 7.5), 60, 96)
        base_etiv = np.clip(np.random.normal(1480, 160), 1120, 1980)
        base_nwbv = np.clip(np.random.normal(0.74, 0.04), 0.65, 0.83)
        base_asf = np.clip(1480.0 / base_etiv, 0.88, 1.55)
        
        # Initial clinical status
        status_prob = np.random.rand()
        if status_prob < 0.50:
            initial_group = "Nondemented"
            base_mmse = int(np.clip(np.random.normal(29.2, 1.0), 26, 30))
            base_cdr = 0.0
        elif status_prob < 0.85:
            initial_group = "Demented"
            base_mmse = int(np.clip(np.random.normal(24.0, 3.5), 15, 28))
            base_cdr = np.random.choice([0.5, 1.0], p=[0.75, 0.25])
        else:
            initial_group = "Converted"
            base_mmse = int(np.clip(np.random.normal(28.5, 1.2), 26, 30))
            base_cdr = 0.0

        n_visits = np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.20, 0.10])
        delays = [0, int(np.random.uniform(400, 750)), int(np.random.uniform(900, 1400)), int(np.random.uniform(1600, 2200))]
        
        for v in range(n_visits):
            mri_id = f"{subj_id}_MR{v+1}"
            visit_num = v + 1
            mr_delay = delays[v]
            age_v = round(base_age + (mr_delay / 365.25), 1)
            
            # Longitudinal brain atrophy & cognitive changes
            years_passed = mr_delay / 365.25
            nwbv_v = round(max(0.60, base_nwbv - (0.006 * years_passed) + np.random.normal(0, 0.003)), 3)
            etiv_v = int(base_etiv + np.random.normal(0, 10))
            asf_v = round(1480.0 / etiv_v, 3)
            
            if initial_group == "Nondemented":
                group_v = "Nondemented"
                mmse_v = int(np.clip(base_mmse - (0.15 * years_passed) + np.random.normal(0, 0.5), 24, 30))
                cdr_v = 0.0
            elif initial_group == "Demented":
                group_v = "Demented"
                mmse_v = int(np.clip(base_mmse - (1.2 * years_passed) + np.random.normal(0, 1.0), 10, 28))
                cdr_v = 1.0 if (base_cdr >= 0.5 and years_passed > 2.0) else base_cdr
            else: # Converted
                if v == 0:
                    group_v = "Converted"
                    mmse_v = base_mmse
                    cdr_v = 0.0
                else:
                    group_v = "Converted"
                    mmse_v = int(np.clip(base_mmse - (1.5 * years_passed) + np.random.normal(0, 1.0), 16, 27))
                    cdr_v = 0.5
            
            records.append({
                "Subject ID": subj_id,
                "MRI ID": mri_id,
                "Group": group_v,
                "Visit": visit_num,
                "MR Delay": mr_delay,
                "M/F": gender,
                "Hand": hand,
                "Age": age_v,
                "EDUC": educ,
                "SES": ses,
                "MMSE": mmse_v,
                "CDR": cdr_v,
                "eTIV": etiv_v,
                "nWBV": nwbv_v,
                "ASF": asf_v,
            })
            
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated OASIS fixture: {output_path} ({len(df)} rows across {n_subjects} subjects)")

def generate_proact_fixture(output_path: Path, n_subjects: int = 80, random_seed: int = 42):
    np.random.seed(random_seed)
    records = []
    
    for i in range(1, n_subjects + 1):
        subj_id = f"ALS_{i:04d}"
        age = int(np.clip(np.random.normal(57.0, 10.5), 28, 82))
        gender = np.random.choice(["Male", "Female"], p=[0.60, 0.40])
        site_onset = np.random.choice(["Limb", "Bulbar"], p=[0.72, 0.28])
        diag_delay = int(np.clip(np.random.exponential(12.0) + 3, 3, 48))
        riluzole = int(np.random.choice([0, 1], p=[0.25, 0.75]))
        
        # Clinical progression profile
        base_rate = np.random.normal(1.0, 0.35)
        if site_onset == "Bulbar":
            base_rate += 0.30
        progression_slope = round(max(0.15, base_rate + np.random.normal(0, 0.1)), 2)
        fast_prog = 1 if progression_slope >= 1.0 else 0
        
        base_alsfrs = int(np.clip(np.random.normal(44, 2.5), 36, 48))
        base_fvc = round(np.clip(np.random.normal(92, 14), 45, 125), 1)
        creatinine = round(np.clip(np.random.normal(1.05, 0.22), 0.5, 1.8), 2)
        albumin = round(np.clip(np.random.normal(4.2, 0.35), 3.2, 5.0), 2)
        alt = int(np.clip(np.random.normal(26, 11), 10, 65))
        
        n_visits = np.random.choice([2, 3, 4], p=[0.30, 0.50, 0.20])
        visit_months = [0, 3, 6, 9][:n_visits]
        
        for vm in visit_months:
            alsfrs_v = int(max(0, round(base_alsfrs - (progression_slope * vm) + np.random.normal(0, 0.8))))
            fvc_v = round(max(20.0, base_fvc - (1.8 * vm) + np.random.normal(0, 2.0)), 1)
            
            records.append({
                "Subject_ID": subj_id,
                "Visit_Month": vm,
                "Age": age,
                "Gender": gender,
                "Site_of_Onset": site_onset,
                "Disease_Duration_Months": diag_delay + vm,
                "ALSFRS_R_Total": alsfrs_v,
                "FVC_percent": fvc_v,
                "Creatinine": creatinine,
                "Albumin": albumin,
                "ALT": alt,
                "Riluzole_Usage": riluzole,
                "Progression_Slope": progression_slope,
                "Fast_Progression": fast_prog,
            })
            
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated PRO-ACT fixture: {output_path} ({len(df)} rows across {n_subjects} subjects)")

def generate_ptbxl_fixture(output_path: Path, n_records: int = 150, random_seed: int = 42):
    np.random.seed(random_seed)
    records = []
    
    diag_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    diag_probs = [0.45, 0.22, 0.15, 0.12, 0.06]
    
    # 100 unique patients for 150 ECGs (some repeated scans)
    patient_ids = [f"PTB_{p:04d}" for p in range(1, 101)]
    
    for i in range(1, n_records + 1):
        pat_id = np.random.choice(patient_ids)
        age = int(np.clip(np.random.normal(62, 14), 20, 94))
        sex = int(np.random.choice([0, 1], p=[0.48, 0.52])) # 0=Female, 1=Male
        diag = np.random.choice(diag_classes, p=diag_probs)
        
        hr = int(np.clip(np.random.normal(74, 13), 45, 130))
        pr = int(np.clip(np.random.normal(160, 22), 105, 260))
        qrs = int(np.clip(np.random.normal(92, 14), 70, 160))
        qt = int(np.clip(np.random.normal(395, 32), 310, 510))
        qtc = int(round(qt / np.sqrt(60.0 / hr)))
        
        records.append({
            "ecg_id": i,
            "patient_id": pat_id,
            "age": age,
            "sex": sex,
            "diagnostic_class": diag,
            "heart_rate": hr,
            "pr_interval": pr,
            "qrs_duration": qrs,
            "qt_interval": qt,
            "qtc_interval": qtc,
            "is_abnormal": 0 if diag == "NORM" else 1,
        })
        
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated PTB-XL fixture: {output_path} ({len(df)} records)")

def generate_parkinsons_telemonitoring_fixture(output_path: Path, n_subjects: int = 20, random_seed: int = 42):
    np.random.seed(random_seed)
    records = []
    
    for subj in range(1, n_subjects + 1):
        age = int(np.clip(np.random.normal(64, 8), 45, 85))
        sex = int(np.random.choice([0, 1], p=[0.35, 0.65]))
        base_motor = np.clip(np.random.normal(24, 7), 10, 42)
        progression_rate = np.clip(np.random.normal(0.04, 0.02), 0.0, 0.12)
        
        n_recordings = np.random.randint(15, 30)
        test_times = np.sort(np.random.uniform(0, 180, n_recordings))
        
        for t in test_times:
            motor = round(float(base_motor + progression_rate * t + np.random.normal(0, 0.5)), 3)
            total = round(float(motor * 1.3 + np.random.normal(0, 0.8)), 3)
            
            jitter_pct = round(max(0.001, np.random.normal(0.005, 0.002)), 5)
            shimmer = round(max(0.005, np.random.normal(0.025, 0.010)), 5)
            nhr = round(max(0.001, np.random.normal(0.02, 0.01)), 6)
            hnr = round(max(10.0, np.random.normal(22.0, 4.0)), 3)
            rpde = round(np.clip(np.random.normal(0.48, 0.08), 0.2, 0.8), 5)
            dfa = round(np.clip(np.random.normal(0.65, 0.07), 0.5, 0.85), 5)
            ppe = round(np.clip(np.random.normal(0.20, 0.06), 0.05, 0.5), 5)
            
            records.append({
                "subject#": subj,
                "age": age,
                "sex": sex,
                "test_time": round(float(t), 4),
                "motor_UPDRS": motor,
                "total_UPDRS": total,
                "Jitter(%)": jitter_pct,
                "Jitter(Abs)": round(jitter_pct * 0.005, 7),
                "Jitter:RAP": round(jitter_pct * 0.55, 5),
                "Jitter:PPQ5": round(jitter_pct * 0.50, 5),
                "Jitter:DDP": round(jitter_pct * 1.65, 5),
                "Shimmer": shimmer,
                "Shimmer(dB)": round(shimmer * 10.0, 3),
                "Shimmer:APQ3": round(shimmer * 0.5, 5),
                "Shimmer:APQ5": round(shimmer * 0.55, 5),
                "Shimmer:APQ11": round(shimmer * 0.75, 5),
                "Shimmer:DDA": round(shimmer * 1.5, 5),
                "NHR": nhr,
                "HNR": hnr,
                "RPDE": rpde,
                "DFA": dfa,
                "PPE": ppe,
            })
            
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated Parkinson's Telemonitoring fixture: {output_path} ({len(df)} records across {n_subjects} subjects)")


if __name__ == "__main__":
    base_dir = Path("data/synthetic")
    generate_oasis_fixture(base_dir / "synthetic_oasis_longitudinal.csv")
    generate_proact_fixture(base_dir / "synthetic_proact_als.csv")
    generate_ptbxl_fixture(base_dir / "synthetic_ptbxl_ecg.csv")
    generate_parkinsons_telemonitoring_fixture(base_dir / "synthetic_parkinsons_telemonitoring.csv")
