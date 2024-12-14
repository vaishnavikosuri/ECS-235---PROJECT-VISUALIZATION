import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class InsiderThreatPreprocessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Initialize with dynamic thresholds that will be set during analysis
        self.risk_thresholds = {
            'low': 0.3,  # Will be updated based on data
            'medium': 0.7 # Will be updated based on data
        }
        self.business_hours = {
            'start': 8,
            'end': 18
        }
        # Feature weights will be calculated dynamically
        self.feature_weights = {
            'off_hours_ratio': 0.2,
            'weekend_ratio': 0.1,
            'suspicious_file_ops': 0.25,
            'device_anomalies': 0.15,
            'resource_usage': 0.15,
            'unique_pc_ratio': 0.15
        }
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate all data sources"""
        try:
            print("Loading logon.csv...")
            logon_df = pd.read_csv(f"{self.data_dir}/logon.csv")
            
            print("Loading device.csv...")
            device_df = pd.read_csv(f"{self.data_dir}/device.csv")
            
            print("Loading file.csv...")
            file_df = pd.read_csv(f"{self.data_dir}/file.csv", 
                                low_memory=False,
                                dtype={'to_removable_media': str, 
                                      'from_removable_media': str})
            
            print("Loading decoy_file.csv...")
            decoy_df = pd.read_csv(f"{self.data_dir}/decoy_file.csv")

            print("Loading LDAP data...")
            ldap_df = self._load_ldap_data()
            
            # Convert boolean columns after loading
            file_df['to_removable_media'] = file_df['to_removable_media'].map({'TRUE': True, 'FALSE': False})
            file_df['from_removable_media'] = file_df['from_removable_media'].map({'TRUE': True, 'FALSE': False})
            
            print(f"\nData loading summary:")
            print(f"- Logon records: {len(logon_df):,}")
            print(f"- Device events: {len(device_df):,}")
            print(f"- File events: {len(file_df):,}")
            print(f"- Decoy files: {len(decoy_df):,}")
            print(f"- LDAP records: {len(ldap_df):,}")
            
            # Validate data quality
            self._validate_data_quality(logon_df, device_df, file_df)
            
            return logon_df, device_df, file_df, decoy_df, ldap_df
            
        except Exception as e:
            print(f"\nError details:")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for files in: {self.data_dir}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

    def _load_ldap_data(self) -> pd.DataFrame:
        """Load and combine all LDAP CSV files"""
        try:
            ldap_path = os.path.join(self.data_dir, 'LDAP')
            if not os.path.exists(ldap_path):
                print(f"Warning: LDAP directory not found at {ldap_path}")
                return pd.DataFrame(columns=['employee_name', 'user_id', 'email', 'role',
                                          'projects', 'business_unit', 'functional_unit',
                                          'department', 'team', 'supervisor'])
            
            ldap_files = glob.glob(os.path.join(ldap_path, '*.csv'))
            if not ldap_files:
                print(f"Warning: No CSV files found in LDAP directory {ldap_path}")
                return pd.DataFrame(columns=['employee_name', 'user_id', 'email', 'role',
                                          'projects', 'business_unit', 'functional_unit',
                                          'department', 'team', 'supervisor'])
            
            ldap_dfs = []
            for file in ldap_files:
                try:
                    df = pd.read_csv(file, 
                                   names=['employee_name', 'user_id', 'email', 'role',
                                         'projects', 'business_unit', 'functional_unit',
                                         'department', 'team', 'supervisor'])
                    ldap_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Error reading LDAP file {file}: {str(e)}")
                    continue
            
            if not ldap_dfs:
                print("Warning: No valid LDAP data could be loaded")
                return pd.DataFrame(columns=['employee_name', 'user_id', 'email', 'role',
                                          'projects', 'business_unit', 'functional_unit',
                                          'department', 'team', 'supervisor'])
            
            return pd.concat(ldap_dfs, ignore_index=True).drop_duplicates(subset=['user_id'])
            
        except Exception as e:
            print(f"Warning: Error in LDAP data loading: {str(e)}")
            return pd.DataFrame(columns=['employee_name', 'user_id', 'email', 'role',
                                      'projects', 'business_unit', 'functional_unit',
                                      'department', 'team', 'supervisor'])
    
    def _validate_data_quality(self, logon_df: pd.DataFrame, device_df: pd.DataFrame, file_df: pd.DataFrame):
        """Validate data quality and completeness"""
        # Check for required columns
        required_columns = {
            'logon': ['user', 'date', 'pc'],
            'device': ['user', 'date', 'activity'],
            'file': ['user', 'date', 'filename', 'activity']
        }
        
        for df, cols in zip([logon_df, device_df, file_df], 
                          [required_columns['logon'], required_columns['device'], required_columns['file']]):
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset and normalize dates"""
        df_cleaned = df.dropna(subset=['user'] if 'user' in df.columns else df.columns)
        
        if 'date' in df_cleaned.columns:
            try:
                df_cleaned['datetime'] = pd.to_datetime(df_cleaned['date'])
                print("\nSample of parsed dates:")
                print(df_cleaned[['date', 'datetime']].head())
            except Exception as e:
                print(f"Warning: Error in date parsing: {str(e)}")
                print(f"Sample date values: {df_cleaned['date'].head().tolist()}")
                raise
        
        return df_cleaned

    def calculate_feature_weights(self, behavior_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature weights with proper validation and error handling
        """
        try:
            # 1. Identify valid numeric features
            numeric_columns = behavior_data.select_dtypes(include=[np.number]).columns
            
            # 2. Initialize weights with default values
            weights = {
                'off_hours_ratio': 0.3,
                'suspicious_file_ops': 0.3,
                'device_anomalies': 0.2,
                'removable_media_ops': 0.1,
                'unique_pc_ratio': 0.1
            }
            
            # 3. Validate if we can compute correlations
            if len(numeric_columns) > 1:
                # Compute correlations with suspicious file operations
                if 'suspicious_file_ops' in numeric_columns:
                    correlations = behavior_data[numeric_columns].corr()['suspicious_file_ops']
                    
                    # Update weights based on correlations
                    for feature in weights.keys():
                        if feature in correlations.index:
                            # Convert correlation to positive weight
                            weights[feature] = abs(correlations[feature])
                    
                    # Normalize weights to sum to 1
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights

        except Exception as e:
            print(f"Warning: Error in weight calculation: {str(e)}")
            print("Using default weights")
            return self.feature_weights
        
    def determine_risk_thresholds(self, risk_scores: pd.Series) -> Dict[str, float]:
        """Determine risk thresholds using statistical analysis"""
        mean = risk_scores.mean()
        std = risk_scores.std()
        
        # Ensure thresholds are properly ordered
        self.risk_thresholds = {
            'low': mean - std,      # Lower threshold
            'medium': mean + std    # Upper threshold
        }
        
        # Validate thresholds are increasing
        if self.risk_thresholds['low'] >= self.risk_thresholds['medium']:
            # If thresholds are invalid, use percentile-based approach
            self.risk_thresholds = {
                'low': risk_scores.quantile(0.33),    # 33rd percentile
                'medium': risk_scores.quantile(0.66)  # 66th percentile
            }
        
        return self.risk_thresholds
    
    def extract_user_features(self, logon_df: pd.DataFrame, device_df: pd.DataFrame, 
                            file_df: pd.DataFrame, decoy_df: pd.DataFrame, 
                            ldap_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive behavioral features"""
        print("\nStarting feature extraction...")
        
        # Get unique users
        users = pd.concat([logon_df['user'], device_df['user'], file_df['user']]).unique()
        print(f"Processing {len(users)} unique users...")
        
        features = []
        for i, user in enumerate(users):
            if i % 100 == 0:  # Progress update
                print(f"Processing user {i}/{len(users)}...")
                
            # Extract all features for user
            user_features = self._compute_user_features(user, logon_df, device_df, file_df)
            features.append(user_features)
        
        features_df = pd.DataFrame(features)
        
        # Calculate optimal feature weights
        self.calculate_feature_weights(features_df)
        
        return features_df

    def _compute_user_features(self, user: str, logon_df: pd.DataFrame, 
                             device_df: pd.DataFrame, file_df: pd.DataFrame) -> Dict:
        """Compute comprehensive features for a single user"""
        # Time-based features
        time_features = self._compute_time_features(
            logon_df[logon_df['user'] == user].copy())
        
        # Device features
        device_features = self._compute_device_features(
            device_df[device_df['user'] == user].copy())
        
        # File operation features
        file_features = self._compute_file_features(
            file_df[file_df['user'] == user].copy())
        
        # Combine all features
        return {
            'user': user,
            **time_features,
            **device_features,
            **file_features
        }

    def _compute_time_features(self, user_data: pd.DataFrame) -> Dict:
        """Compute time-based behavioral features"""
        if len(user_data) == 0:
            return self._get_default_time_features()

        # Calculate hour and weekend features
        user_data.loc[:, 'hour'] = user_data['datetime'].dt.hour
        user_data.loc[:, 'is_weekend'] = user_data['datetime'].dt.dayofweek >= 5
        
        # Calculate off-hours activity
        off_hours = ((user_data['hour'] < self.business_hours['start']) | 
                    (user_data['hour'] >= self.business_hours['end']))
        
        return {
            'total_events': len(user_data),
            'off_hours_ratio': off_hours.mean(),
            'weekend_ratio': user_data['is_weekend'].mean(),
            'unique_pc_count': user_data['pc'].nunique(),
            'hour_variance': user_data['hour'].std() if len(user_data) > 1 else 0
        }

    def _compute_device_features(self, user_data: pd.DataFrame) -> Dict:
        """Compute device-related behavioral features"""
        if len(user_data) == 0:
            return self._get_default_device_features()

        connects = len(user_data[user_data['activity'] == 'Connect'])
        disconnects = len(user_data[user_data['activity'] == 'Disconnect'])
        
        return {
            'device_connects': connects,
            'device_disconnects': disconnects,
            'missing_disconnects': max(0, connects - disconnects),
            'device_anomalies': self._detect_device_anomalies(user_data)
        }

    def _compute_file_features(self, user_data: pd.DataFrame) -> Dict:
        """Compute file operation behavioral features"""
        if len(user_data) == 0:
            return self._get_default_file_features()

        suspicious_activities = ['File Delete', 'File Copy']
        
        return {
            'file_operations': len(user_data),
            'suspicious_file_ops': len(user_data[user_data['activity'].isin(suspicious_activities)]),
            'removable_media_ops': (user_data['to_removable_media'].sum() + 
                                  user_data['from_removable_media'].sum()),
            'resource_usage': len(user_data['filename'].unique())
        }

    def _detect_device_anomalies(self, user_data: pd.DataFrame) -> int:
        """Detect anomalies in device usage patterns"""
        if len(user_data) < 10:  # Not enough data for meaningful anomaly detection
            return 0
            
        # Create feature vector for anomaly detection
        features = pd.get_dummies(user_data['activity'])
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        return sum(anomalies == -1)  # Count anomalies

    def compute_risk_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk scores with proper error handling and data validation
        """
        print("\nComputing risk scores...")
        
        # Create a copy to avoid modifications
        df = features_df.copy()
        
        try:
            # 1. Select and validate numeric features
            numeric_features = []
            for feature in self.feature_weights.keys():
                if feature in df.columns:
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        # Check for NaN values
                        if df[feature].isna().any():
                            print(f"Warning: NaN values found in {feature}, filling with 0")
                            df[feature] = df[feature].fillna(0)
                        numeric_features.append(feature)
                    else:
                        print(f"Warning: Feature {feature} is not numeric, skipping")

            if not numeric_features:
                raise ValueError("No valid numeric features found for risk scoring")

            # 2. Normalize features using RobustScaler (handles outliers better)
            for feature in numeric_features:
                if df[feature].std() > 0:  # Only normalize if there's variation
                    df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
                else:
                    df[feature] = 0  # Set to 0 if no variation

            # 3. Calculate initial risk score
            df['risk_score'] = 0
            for feature in numeric_features:
                if feature in self.feature_weights:
                    df['risk_score'] += self.feature_weights[feature] * df[feature]

            # 4. Validate risk scores
            if df['risk_score'].isna().any():
                print("Warning: NaN values in risk scores, filling with minimum")
                min_score = df['risk_score'].min()
                df['risk_score'] = df['risk_score'].fillna(min_score)

            # 5. Normalize risk scores to 0-1 range
            score_min = df['risk_score'].min()
            score_max = df['risk_score'].max()
            if score_max > score_min:
                df['risk_score'] = (df['risk_score'] - score_min) / (score_max - score_min)
            else:
                df['risk_score'] = 0.5  # Default middle value if no variation

            # 6. Assign risk levels
            total_users = len(df)
            low_threshold = df['risk_score'].quantile(0.7)  # Top 30% are medium or high
            high_threshold = df['risk_score'].quantile(0.9)  # Top 10% are high

            df['risk_level'] = 'low'
            df.loc[df['risk_score'] > low_threshold, 'risk_level'] = 'medium'
            df.loc[df['risk_score'] > high_threshold, 'risk_level'] = 'high'

            # 7. Print distribution information
            print("\nRisk Score Distribution:")
            print(f"Low Risk: {len(df[df['risk_level'] == 'low'])} users")
            print(f"Medium Risk: {len(df[df['risk_level'] == 'medium'])} users")
            print(f"High Risk: {len(df[df['risk_level'] == 'high'])} users")

            return df

        except Exception as e:
            print(f"Error in risk score computation: {str(e)}")
            print("Traceback:", traceback.format_exc())
            raise

    def _get_default_time_features(self) -> Dict:
        """Return default values for time features"""
        return {
            'total_events': 0,
            'off_hours_ratio': 0,
            'weekend_ratio': 0,
            'unique_pc_count': 0,
            'hour_variance': 0
        }

    def _get_default_device_features(self) -> Dict:
        """Return default values for device features"""
        return {
            'device_connects': 0,
            'device_disconnects': 0,
            'missing_disconnects': 0,
            'device_anomalies': 0
        }

    def _get_default_file_features(self) -> Dict:
        """Return default values for file features"""
        return {
            'file_operations': 0,
            'suspicious_file_ops': 0,
            'removable_media_ops': 0,
            'resource_usage': 0
        }

def analyze_behavioral_patterns(results_df: pd.DataFrame):
    """Analyze and report detailed behavioral patterns"""
    
    # 1. Off-Hours Activity Analysis
    off_hours_stats = {
        'mean': results_df['off_hours_ratio'].mean(),
        'std': results_df['off_hours_ratio'].std(),
        'max': results_df['off_hours_ratio'].max(),
        'high_risk_mean': results_df[results_df['risk_level'] == 'high']['off_hours_ratio'].mean()
    }
    
    print("\n1. Off-Hours Activity Patterns:")
    print(f"- Average off-hours activity: {off_hours_stats['mean']:.2%}")
    print(f"- High-risk users off-hours average: {off_hours_stats['high_risk_mean']:.2%}")
    print("Risk Indication: Explicit data shows high-risk users have {:.1f}x more off-hours activity"
        .format(off_hours_stats['high_risk_mean'] / off_hours_stats['mean']))

    # 2. File Operation Analysis
    file_ops_stats = {
        'mean': results_df['suspicious_file_ops'].mean(),
        'std': results_df['suspicious_file_ops'].std(),
        'max': results_df['suspicious_file_ops'].max(),
        'high_risk_mean': results_df[results_df['risk_level'] == 'high']['suspicious_file_ops'].mean()
    }
    
    print("\n2. Suspicious File Operations:")
    print(f"- Average suspicious operations: {file_ops_stats['mean']:.1f}")
    print(f"- High-risk users average: {file_ops_stats['high_risk_mean']:.1f}")
    print("Risk Indication: Data shows high-risk users perform {:.1f}x more suspicious file operations"
        .format(file_ops_stats['high_risk_mean'] / file_ops_stats['mean']))

    # 3. Device Usage Patterns
    device_stats = {
        'mean': results_df['device_anomalies'].mean(),
        'high_risk_mean': results_df[results_df['risk_level'] == 'high']['device_anomalies'].mean(),
        'missing_disconnects_avg': results_df['missing_disconnects'].mean(),
        'high_risk_disconnects': results_df[results_df['risk_level'] == 'high']['missing_disconnects'].mean()
    }
    
    print("\n3. Device Usage Patterns:")
    print(f"- Average device anomalies: {device_stats['mean']:.1f}")
    print(f"- High-risk users anomalies: {device_stats['high_risk_mean']:.1f}")
    print(f"- Average missing disconnects: {device_stats['missing_disconnects_avg']:.1f}")
    print("Risk Indication: High-risk users show {:.1f}x more device anomalies"
        .format(device_stats['high_risk_mean'] / device_stats['mean']))

    # 4. Combined Behavior Analysis
    high_risk_users = results_df[results_df['risk_level'] == 'high']
    
    print("\n4. High-Risk Behavior Combinations:")
    for _, user in high_risk_users.iterrows():
        print(f"\nUser {user['user']} Risk Pattern:")
        print(f"- Off-hours activity: {user['off_hours_ratio']:.2%}")
        print(f"- Suspicious file operations: {user['suspicious_file_ops']}")
        print(f"- Device anomalies: {user['device_anomalies']}")
        print(f"- Missing device disconnects: {user['missing_disconnects']}")
        
        # Analyze risk factors
        risk_factors = []
        if user['off_hours_ratio'] > off_hours_stats['mean'] + off_hours_stats['std']:
            risk_factors.append("Excessive off-hours activity")
        if user['suspicious_file_ops'] > file_ops_stats['mean'] + file_ops_stats['std']:
            risk_factors.append("High volume of suspicious file operations")
        if user['device_anomalies'] > device_stats['mean'] + results_df['device_anomalies'].std():
            risk_factors.append("Unusual device usage patterns")
            
        print("Primary Risk Factors:", ", ".join(risk_factors))

def generate_analysis_report(results_df: pd.DataFrame, output_file: str = "insider_threat_report.txt"):
    """Generate a comprehensive analysis report with proper formatting"""
    try:
        with open(output_file, 'w') as f:
            # 1. Overall Risk Distribution
            f.write("\n" + "="*50 + "\n")
            f.write("INSIDER THREAT DETECTION ANALYSIS\n")
            f.write("="*50 + "\n\n")

            # Risk Distribution Table
            risk_dist = results_df['risk_level'].value_counts()
            total_users = len(results_df)
            
            f.write("\n1. RISK DISTRIBUTION\n" + "-"*20 + "\n")
            f.write(f"{'Risk Level':<15}{'Count':>10}{'Percentage':>15}\n")
            f.write("-"*40 + "\n")
            for level in ['high', 'medium', 'low']:
                count = risk_dist.get(level, 0)
                percentage = (count/total_users) * 100
                f.write(f"{level:<15}{count:>10}{percentage:>14.1f}%\n")

            # 2. High Risk Users Analysis
            f.write("\n2. HIGH RISK USERS ANALYSIS\n" + "-"*30 + "\n")
            high_risk = results_df[results_df['risk_level'] == 'high'].nlargest(10, 'risk_score')
            
            # Format table headers
            headers = ['User ID', 'Risk Score', 'Off-Hours%', 'Susp.Files', 'Miss.Discon']
            f.write(f"\n{'':3}".join(f"{h:<15}" for h in headers) + "\n")
            f.write("-"*75 + "\n")
            
            # Write user data
            for _, user in high_risk.iterrows():
                user_line = [
                    f"{user['user']:<15}",
                    f"{user['risk_score']:>9.2f}    ",
                    f"{user['off_hours_ratio']*100:>6.1f}%     ",
                    f"{int(user['suspicious_file_ops']):>6}     ",
                    f"{int(user['missing_disconnects']):>4}"
                ]
                f.write("".join(user_line) + "\n")

            # 3. Behavioral Analysis
            f.write("\n3. BEHAVIORAL PATTERNS\n" + "-"*25 + "\n")
            
            # Calculate statistics
            high_risk = results_df[results_df['risk_level'] == 'high']
            normal = results_df[results_df['risk_level'] == 'low']
            
            patterns = {
                'Off-Hours Access': {
                    'high_risk_avg': high_risk['off_hours_ratio'].mean() * 100,
                    'normal_avg': normal['off_hours_ratio'].mean() * 100,
                    'multiplier': (high_risk['off_hours_ratio'].mean() / 
                                 max(normal['off_hours_ratio'].mean(), 0.001))
                },
                'Suspicious Files': {
                    'high_risk_avg': high_risk['suspicious_file_ops'].mean(),
                    'normal_avg': normal['suspicious_file_ops'].mean(),
                    'multiplier': (high_risk['suspicious_file_ops'].mean() / 
                                 max(normal['suspicious_file_ops'].mean(), 0.001))
                },
                'Device Anomalies': {
                    'high_risk_avg': high_risk['missing_disconnects'].mean(),
                    'normal_avg': normal['missing_disconnects'].mean(),
                    'multiplier': (high_risk['missing_disconnects'].mean() / 
                                 max(normal['missing_disconnects'].mean(), 0.001))
                }
            }
            
            # Write pattern analysis
            f.write(f"{'Behavior Type':<20}{'High Risk':>12}{'Normal':>12}{'Multiple':>12}\n")
            f.write("-"*56 + "\n")
            
            for behavior, stats in patterns.items():
                f.write(f"{behavior:<20}{stats['high_risk_avg']:>11.1f}{stats['normal_avg']:>12.1f}" +
                       f"{stats['multiplier']:>11.1f}x\n")

            # 4. Key Findings
            f.write("\n4. KEY FINDINGS\n" + "-"*15 + "\n")
            
            # Format findings based on statistics
            findings = []
            if patterns['Off-Hours Access']['multiplier'] > 1.5:
                findings.append(f"Off-hours access is {patterns['Off-Hours Access']['multiplier']:.1f}x " +
                             "higher in high-risk users")
            if patterns['Suspicious Files']['multiplier'] > 1.5:
                findings.append(f"Suspicious file operations are {patterns['Suspicious Files']['multiplier']:.1f}x " +
                             "more frequent in high-risk users")
            if patterns['Device Anomalies']['multiplier'] > 1.5:
                findings.append(f"Device anomalies are {patterns['Device Anomalies']['multiplier']:.1f}x " +
                             "more common in high-risk users")
            
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")

        return output_file

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise
    
def main():
    """Main execution function"""
    preprocessor = InsiderThreatPreprocessor(data_dir=".")
    
    try:
        # Load and process data
        print("Loading data...")
        logon_df, device_df, file_df, decoy_df, ldap_df = preprocessor.load_all_data()
        
        print("\nCleaning data...")
        logon_df_cleaned = preprocessor.clean_data(logon_df)
        device_df_cleaned = preprocessor.clean_data(device_df)
        file_df_cleaned = preprocessor.clean_data(file_df)
        
        print("\nExtracting features...")
        features_df = preprocessor.extract_user_features(
            logon_df_cleaned, device_df_cleaned, file_df_cleaned, decoy_df, ldap_df
        )
        
        print("\nComputing risk scores...")
        results_df = preprocessor.compute_risk_scores(features_df)
        
        # Generate analysis report
        print("\nGenerating analysis report...")
        report_file = generate_analysis_report(results_df)
        print(f"\nDetailed analysis saved to: {report_file}")
        
        # Save raw results
        results_df.to_csv("insider_threat_analysis_results.csv", index=False)
        print("Raw results saved to: insider_threat_analysis_results.csv")
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        raise
if __name__ == "__main__":
    main()