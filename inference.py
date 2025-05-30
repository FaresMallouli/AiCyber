import pandas as pd
import numpy as np
import os
import joblib
import warnings
import ipaddress
import json
import requests
from functools import lru_cache
import time # For timestamping predictions

# --- NFStream Import ---
try:
    from nfstream import NFStreamer, NFPlugin
    NFSTREAM_AVAILABLE = True
except ImportError:
    NFSTREAM_AVAILABLE = False
    print("CRITICAL: NFStream library not found. Please install it: pip install nfstream")
    exit()

# --- SHAP Import ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP library not found. Explainability will be disabled.")

# --- IntervalTree (for ASN lookup performance) ---
try:
    from intervaltree import Interval, IntervalTree
    intervaltree_available = True
except ImportError:
    intervaltree_available = False
    print("WARNING: 'intervaltree' library not found. ASN lookup will be VERY SLOW.")

# --- Configuration: Paths and Model Tags ---
BASE_ARTIFACT_PATH = "models" # Where the subdirectories reside

# Model specific sub-paths
MAIN_MODEL_DIR = os.path.join(BASE_ARTIFACT_PATH, "main_threat_detector")
OPERATOR_MODEL_DIR = os.path.join(BASE_ARTIFACT_PATH, "operator_id_model")
SOURCE_CLASSIFIER_DIR = os.path.join(BASE_ARTIFACT_PATH, "source_category_classifier")

# Tags from your training/zipping script
MAIN_FINAL_SAVE_TAG = "rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS" # For main model

# Paths for enrichment data
ENRICHMENT_DATA_BASE_PATH = "external_data"
GEOLITE_ASN_BLOCKS_IPV4_CSV_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "GeoLite2-ASN-Blocks-IPv4.csv")
AWS_RANGES_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "amazon.json")
AZURE_RANGES_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "ServiceTags_Public_20250505.json")
GCP_RANGES_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "gcp.json")
VPN_LIST_NORDVPN_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "NordVPN-Server-IP-List.txt")
TOR_EXIT_NODES_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "tor-exit-nodes.lst")
TOR_NODES_PATH = os.path.join(ENRICHMENT_DATA_BASE_PATH, "tor-nodes.lst")

# NFStream source will be taken as input
# NFSTREAM_SOURCE = 'Realtek RTL8852BE WiFi 6 802.11ax PCIe Adapter' # OLD HARDCODED VALUE

# Other Config
WEB_ATTACK_CLASSES_INF = ['Web_Brute_Force', 'Web_SQL_Injection', 'Web_XSS']
WEB_ATTACK_REFINEMENT_THRESHOLD_INF = 0.7
SHAP_TOP_N_FEATURES = 5 # Number of top SHAP features to print

# --- Global Variables for Enrichment Data ---
ASN_LOOKUP_STRUCTURE_GLOBAL = None
aws_nets_frozen_inf = frozenset()
azure_nets_frozen_inf = frozenset()
gcp_nets_frozen_inf = frozenset()
vpn_nets_frozen_inf = frozenset()
tor_nets_frozen_inf = frozenset()

# --- NFStream Compatible Features List (Superset for ALL models) ---
operator_v2_features = [
    'duration', 'packets_count', 'fwd_packets_count', 'bwd_packets_count',
    'total_payload_bytes', 'fwd_total_payload_bytes', 'bwd_total_payload_bytes',
    'payload_bytes_mean', 'payload_bytes_std', 'payload_bytes_variance',
    'fwd_payload_bytes_mean', 'fwd_payload_bytes_std', 'fwd_payload_bytes_variance',
    'bwd_payload_bytes_mean', 'bwd_payload_bytes_std', 'bwd_payload_bytes_variance',
    'payload_bytes_max', 'payload_bytes_min', 'fwd_payload_bytes_max', 'fwd_payload_bytes_min',
    'bwd_payload_bytes_max', 'bwd_payload_bytes_min', 'fwd_init_win_bytes', 'bwd_init_win_bytes',
    'active_min', 'active_max', 'active_mean', 'active_std', 'idle_min', 'idle_max', 'idle_mean', 'idle_std',
    'bytes_rate', 'packets_rate', 'fwd_packets_rate', 'bwd_packets_rate', 'down_up_rate',
    'fin_flag_counts', 'psh_flag_counts', 'urg_flag_counts', 'ece_flag_counts', 'syn_flag_counts', 'ack_flag_counts', 'cwr_flag_counts', 'rst_flag_counts',
    'fwd_fin_flag_counts', 'fwd_psh_flag_counts', 'fwd_urg_flag_counts', 'fwd_syn_flag_counts', 'fwd_ack_flag_counts', 'fwd_rst_flag_counts',
    'bwd_fin_flag_counts', 'bwd_psh_flag_counts', 'bwd_urg_flag_counts', 'bwd_syn_flag_counts', 'bwd_ack_flag_counts', 'bwd_rst_flag_counts',
    'packets_IAT_mean', 'packet_IAT_std', 'packet_IAT_max', 'packet_IAT_min', 'packet_IAT_total',
    'fwd_packets_IAT_mean', 'fwd_packets_IAT_std', 'fwd_packets_IAT_max', 'fwd_packets_IAT_min', 'fwd_packets_IAT_total',
    'bwd_packets_IAT_mean', 'bwd_packets_IAT_std', 'bwd_packets_IAT_max', 'bwd_packets_IAT_min', 'bwd_packets_IAT_total',
    'subflow_fwd_packets', 'subflow_bwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_bytes'
]
main_model_extra_features = [
    'bwd_mean_header_bytes', 'fwd_min_header_bytes', 'bwd_avg_segment_size',
    'bwd_std_header_bytes', 'mean_header_bytes', 'min_header_bytes',
    'total_header_bytes', 'bwd_max_header_bytes', 'fwd_avg_segment_size',
    'std_header_bytes', 'max_header_bytes', 'fwd_mean_header_bytes',
    'bwd_min_header_bytes', 'fwd_max_header_bytes', 'avg_segment_size',
    'fwd_std_header_bytes'
]
NFSTREAM_COMPATIBLE_FEATURES = sorted(list(set(operator_v2_features + main_model_extra_features)))
NFSTREAM_COMPATIBLE_FEATURES = [f.strip().replace(' ', '_').replace('/', '_').replace('-', '_') for f in NFSTREAM_COMPATIBLE_FEATURES]

# --- Enrichment Functions (Same as before) ---
def parse_ip_network_inf(ip_str):
    try: return ipaddress.ip_network(ip_str, strict=False)
    except ValueError: return None
def load_asn_data_optimized_inf(asn_blocks_path):
    global intervaltree_available; print(f"Loading ASN Blocks from {asn_blocks_path} for inference...")
    if not asn_blocks_path or not os.path.exists(asn_blocks_path): print("Error: ASN Blocks file not found."); return None
    if intervaltree_available:
        try:
            asn_tree = IntervalTree(); count = 0
            with open(asn_blocks_path, 'r', encoding='utf-8') as f:
                next(f);
                for line in f:
                    parts = line.strip().split(',', 2);
                    if len(parts) != 3: continue
                    network_cidr, asn_num_str, asn_org_str = parts[0], parts[1], parts[2].strip('"')
                    if not asn_num_str or not network_cidr: continue
                    network = parse_ip_network_inf(network_cidr)
                    if network and network.version == 4:
                        start_ip_int = int(network.network_address); end_ip_int = int(network.broadcast_address) + 1
                        asn_num = int(asn_num_str); asn_org = asn_org_str if asn_org_str else 'Unknown Org'
                        asn_tree[start_ip_int:end_ip_int] = (asn_num, asn_org); count += 1
            print(f"Loaded {count} IPv4 ASN blocks into IntervalTree."); return asn_tree
        except Exception as e: print(f"Error loading ASN Blocks with IntervalTree: {e}")
    print("ASN data could not be loaded effectively."); return None
def load_cloud_ranges_inf(provider, file_path=None, url=None):
    print(f"Loading {provider} IP ranges..."); ranges_json = None
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f: ranges_json = json.load(f)
        except Exception as e: print(f"  Error reading local {provider} file: {e}")
    elif url:
        try:
            response = requests.get(url, timeout=10); response.raise_for_status(); ranges_json = response.json()
        except Exception as e_download: print(f"  Error downloading {provider} ranges: {e_download}")
    if not ranges_json: return frozenset()
    networks = set()
    try:
        if provider == 'AWS':
            for prefix in ranges_json.get('prefixes', []):
                 if prefix.get('ip_prefix'): net = parse_ip_network_inf(prefix['ip_prefix']); networks.add(net) if net and net.version == 4 else None
        elif provider == 'Azure':
            for value in ranges_json.get('values', []):
                for prefix_item in value.get('properties', {}).get('addressPrefixes', []):
                    net = parse_ip_network_inf(prefix_item); networks.add(net) if net and net.version == 4 else None
        elif provider == 'GCP':
             for entry in ranges_json.get('prefixes', []):
                 if 'ipv4Prefix' in entry: net = parse_ip_network_inf(entry['ipv4Prefix']); networks.add(net) if net and net.version == 4 else None
        networks.discard(None); return frozenset(networks)
    except Exception: return frozenset()
def load_ip_list_from_multiple_files_inf(file_paths_list):
    combined_ips_or_nets = set()
    for file_path in file_paths_list:
        if not file_path or not os.path.exists(file_path): continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'): net = parse_ip_network_inf(line);
                    if net: combined_ips_or_nets.add(net)
            combined_ips_or_nets.discard(None)
        except Exception: pass
    return frozenset(combined_ips_or_nets)
@lru_cache(maxsize=2**16)
def get_asn_info_for_ip_cached_inf(ip_str):
    global ASN_LOOKUP_STRUCTURE_GLOBAL;
    if ASN_LOOKUP_STRUCTURE_GLOBAL is None: return -1, 'ASN DB Not Loaded'
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.version != 4: return -1, 'Non-IPv4 Address'
        if intervaltree_available and isinstance(ASN_LOOKUP_STRUCTURE_GLOBAL, IntervalTree):
            ip_int = int(ip_obj); intervals = ASN_LOOKUP_STRUCTURE_GLOBAL[ip_int]
            if intervals:
                interval_data = min(intervals, key=lambda i: i.end - i.begin).data
                asn_num, asn_org = interval_data
                return int(asn_num) if asn_num is not None else -1, asn_org if asn_org else "Unknown Org"
        return -1, 'ASN Lookup Error'
    except ValueError: return -1, 'Invalid IP String for ASN'
    except Exception: return -1, 'ASN Lookup Exception'
@lru_cache(maxsize=2**16)
def check_ip_membership_inf(ip_str, aws_nets_f, azure_nets_f, gcp_nets_f, vpn_nets_f, tor_nets_f):
     try: ip_obj = ipaddress.ip_address(ip_str)
     except ValueError: return False, False, False, False, False
     if ip_obj.version != 4: return False, False, False, False, False
     is_aws = any(ip_obj in net for net in aws_nets_f); is_azure = any(ip_obj in net for net in azure_nets_f) if not is_aws else False
     is_gcp = any(ip_obj in net for net in gcp_nets_f) if not (is_aws or is_azure) else False
     is_vpn = any(ip_obj in net for net in vpn_nets_f); is_tor = any(ip_obj in net for net in tor_nets_f)
     return is_aws, is_azure, is_gcp, is_vpn, is_tor
def categorize_source_inf(asn_org_str, is_cloud, is_proxy):
    if is_proxy: return 'VPN/Proxy';
    if is_cloud: return 'Cloud'
    if pd.isna(asn_org_str) or not isinstance(asn_org_str, str) or "Unknown" in asn_org_str or "Error" in asn_org_str or not asn_org_str: return 'Unknown'
    org_lower = asn_org_str.lower()
    hosting_kws = ['hosting', 'host', 'ovh', 'linode', 'digitalocean', 'hetzner', 'server', 'data center', 'datacenter', 'online sas', 'leaseweb', 'contabo', 'dedicated', 'colocation', 'cdn', 'vultr', 'equinix', 'intergenia', 'ionos', 'godaddy', 'google cloud', 'amazon technologies', 'microsoft corporation']
    edu_kws = ['university', 'college', 'school', 'institute of technology', 'research', 'education', 'academic', 'national laboratory', 'bibliothek', 'universitat', 'ecole', 'universidad', 'renater']
    isp_kws = ['isp', 'internet', 'telecom', 'communication', 'cable', 'broadband', 'mobile', 'residential', 'verizon', 'comcast', 'at&t', 'centurylink', 'spectrum', 'telefonica', 'deutsche telekom', 'orange', 'telia', 'vodafone', 'telstra', 'bt ', 'sky ', 'telus', 'bell canada', 'rogers', 'reliance jio', 'bharti airtel', 'telemar', 'optus', 'swisscom', 'liberty global', 'tim ', 'claro', 'charter ', 'cox ', 'sprint', 't-mobile']
    business_kws = ['inc', 'ltd', 'llc', 'corp', 'corporation', 'group', 'gmbh', 's.a', 'l.p.', 'limited', 'plc', 'associates', 'bank', 'financial', 'versicherung', 'ag', 'b.v.', 's.p.a.', 'holdings', 'llp']
    if any(kw in org_lower for kw in hosting_kws): return 'Hosting/DataCenter'
    if any(kw in org_lower for kw in edu_kws): return 'Education'
    if any(kw in org_lower for kw in isp_kws): return 'ISP/Residential'
    if any(kw in org_lower for kw in business_kws): return 'Other/Business'
    return 'Other/Business'

# --- Model Loading ---
MODELS = {}
def load_all_artifacts_inference():
    global MODELS, ASN_LOOKUP_STRUCTURE_GLOBAL, aws_nets_frozen_inf, azure_nets_frozen_inf, gcp_nets_frozen_inf, vpn_nets_frozen_inf, tor_nets_frozen_inf
    print("Loading all models and artifacts for inference...")
    warnings.filterwarnings("ignore")
    try:
        MODELS['main_detector'] = joblib.load(os.path.join(MAIN_MODEL_DIR, f"main_detector_{MAIN_FINAL_SAVE_TAG}.joblib"))
        MODELS['web_detector'] = joblib.load(os.path.join(MAIN_MODEL_DIR, f"web_detector_{MAIN_FINAL_SAVE_TAG}.joblib"))
        MODELS['main_scaler'] = joblib.load(os.path.join(MAIN_MODEL_DIR, f"main_model_scaler_{MAIN_FINAL_SAVE_TAG}.joblib"))
        MODELS['main_label_encoder'] = joblib.load(os.path.join(MAIN_MODEL_DIR, f"label_encoder_{MAIN_FINAL_SAVE_TAG}.joblib"))
        MODELS['main_selected_features'] = joblib.load(os.path.join(MAIN_MODEL_DIR, f"selected_features_main_model_{MAIN_FINAL_SAVE_TAG}.joblib"))
        if SHAP_AVAILABLE: MODELS['main_rf_explainer'] = shap.Explainer(MODELS['main_detector'])
        print("  Main Threat Detector artifacts loaded.")

        op_fs_tag_v2 = "_FS_True"
        operator_model_filename_v2 = f"operator_id_model_cpu_sklearn_rf{op_fs_tag_v2}.joblib"
        operator_scaler_filename_v2 = f"operator_id_scaler_cpu_sklearn_rf{op_fs_tag_v2}.joblib"
        operator_selected_features_filename_v2 = "operator_id_selected_features.joblib"
        MODELS['operator_model'] = joblib.load(os.path.join(OPERATOR_MODEL_DIR, operator_model_filename_v2))
        MODELS['operator_scaler'] = joblib.load(os.path.join(OPERATOR_MODEL_DIR, operator_scaler_filename_v2))
        MODELS['operator_selected_features'] = joblib.load(os.path.join(OPERATOR_MODEL_DIR, operator_selected_features_filename_v2))
        if SHAP_AVAILABLE: MODELS['operator_rf_explainer'] = shap.Explainer(MODELS['operator_model'])
        print(f"  Operator ID Model artifacts loaded (V2: {operator_model_filename_v2}).")

        MODELS['source_classifier_pipeline'] = joblib.load(os.path.join(SOURCE_CLASSIFIER_DIR, "source_classifier_pipeline.joblib"))
        MODELS['source_classifier_label_encoder'] = joblib.load(os.path.join(SOURCE_CLASSIFIER_DIR, "source_classifier_label_encoder.joblib"))
        print("  Source Classifier artifacts loaded.")

        print("  Loading enrichment databases..."); ASN_LOOKUP_STRUCTURE_GLOBAL = load_asn_data_optimized_inf(GEOLITE_ASN_BLOCKS_IPV4_CSV_PATH)
        aws_nets_frozen_inf = load_cloud_ranges_inf('AWS', AWS_RANGES_PATH); azure_nets_frozen_inf = load_cloud_ranges_inf('Azure', AZURE_RANGES_PATH)
        gcp_nets_frozen_inf = load_cloud_ranges_inf('GCP', GCP_RANGES_PATH)
        vpn_files_to_load = [p for p in [VPN_LIST_NORDVPN_PATH] if p and os.path.exists(p)]; vpn_nets_frozen_inf = load_ip_list_from_multiple_files_inf(vpn_files_to_load)
        tor_files_to_load = [p for p in [TOR_EXIT_NODES_PATH, TOR_NODES_PATH] if p and os.path.exists(p)]; tor_nets_frozen_inf = load_ip_list_from_multiple_files_inf(tor_files_to_load)
        print("  Enrichment databases loaded/attempted.\nAll artifacts loaded successfully.")
        return True
    except FileNotFoundError as e: print(f"CRITICAL ERROR loading artifact: {e}. Check paths."); return False
    except Exception as e: print(f"Unexpected error loading artifacts: {e}"); import traceback; traceback.print_exc(); return False

# --- Feature Extraction from NFStream Flow ---
def nfstream_flow_to_features_dict(flow, expected_feature_list):
    features = {}
    def get_attr(obj, attr_name, default=0.0):
        val = getattr(obj, attr_name, default)
        return default if val is None else float(val)
    features['duration'] = get_attr(flow, 'duration'); features['packets_count'] = get_attr(flow, 'bidirectional_packets')
    features['fwd_packets_count'] = get_attr(flow, 'src2dst_packets'); features['bwd_packets_count'] = get_attr(flow, 'dst2src_packets')
    features['total_payload_bytes'] = get_attr(flow, 'bidirectional_payload_bytes'); features['fwd_total_payload_bytes'] = get_attr(flow, 'src2dst_payload_bytes')
    features['bwd_total_payload_bytes'] = get_attr(flow, 'dst2src_payload_bytes'); features['payload_bytes_mean'] = get_attr(flow, 'bidirectional_mean_payload_bytes')
    features['payload_bytes_std'] = get_attr(flow, 'bidirectional_stddev_payload_bytes'); features['payload_bytes_variance'] = get_attr(flow, 'bidirectional_stddev_payload_bytes')**2
    features['fwd_payload_bytes_mean'] = get_attr(flow, 'src2dst_mean_payload_bytes'); features['fwd_payload_bytes_std'] = get_attr(flow, 'src2dst_stddev_payload_bytes')
    features['fwd_payload_bytes_variance'] = get_attr(flow, 'src2dst_stddev_payload_bytes')**2; features['bwd_payload_bytes_mean'] = get_attr(flow, 'dst2src_mean_payload_bytes')
    features['bwd_payload_bytes_std'] = get_attr(flow, 'dst2src_stddev_payload_bytes'); features['bwd_payload_bytes_variance'] = get_attr(flow, 'dst2src_stddev_payload_bytes')**2
    features['payload_bytes_max'] = get_attr(flow, 'bidirectional_max_payload_bytes'); features['payload_bytes_min'] = get_attr(flow, 'bidirectional_min_payload_bytes')
    features['fwd_payload_bytes_max'] = get_attr(flow, 'src2dst_max_payload_bytes'); features['fwd_payload_bytes_min'] = get_attr(flow, 'src2dst_min_payload_bytes')
    features['bwd_payload_bytes_max'] = get_attr(flow, 'dst2src_max_payload_bytes'); features['bwd_payload_bytes_min'] = get_attr(flow, 'dst2src_min_payload_bytes')
    features['fwd_init_win_bytes'] = get_attr(flow, 'src2dst_initial_tcp_window_size'); features['bwd_init_win_bytes'] = get_attr(flow, 'dst2src_initial_tcp_window_size')
    features['active_min'] = get_attr(flow, 'bidirectional_min_active_time'); features['active_max'] = get_attr(flow, 'bidirectional_max_active_time')
    features['active_mean'] = get_attr(flow, 'bidirectional_mean_active_time'); features['active_std'] = get_attr(flow, 'bidirectional_stddev_active_time')
    features['idle_min'] = get_attr(flow, 'bidirectional_min_idle_time'); features['idle_max'] = get_attr(flow, 'bidirectional_max_idle_time')
    features['idle_mean'] = get_attr(flow, 'bidirectional_mean_idle_time'); features['idle_std'] = get_attr(flow, 'bidirectional_stddev_idle_time')
    duration_s = get_attr(flow, 'duration')
    if duration_s > 0:
        features['bytes_rate'] = get_attr(flow, 'bidirectional_bytes') / duration_s; features['packets_rate'] = get_attr(flow, 'bidirectional_packets') / duration_s
        features['fwd_packets_rate'] = get_attr(flow, 'src2dst_packets') / duration_s; features['bwd_packets_rate'] = get_attr(flow, 'dst2src_packets') / duration_s
    else:
        for rate_feat in ['bytes_rate', 'packets_rate', 'fwd_packets_rate', 'bwd_packets_rate']: features[rate_feat] = 0.0
    features['down_up_rate'] = (get_attr(flow, 'dst2src_packets') / get_attr(flow, 'src2dst_packets')) if get_attr(flow, 'src2dst_packets') > 0 else 0.0
    flags_map = {'fin': 'FIN', 'syn': 'SYN', 'rst': 'RST', 'psh': 'PSH', 'ack': 'ACK', 'urg': 'URG', 'ece': 'ECE', 'cwr': 'CWR'}
    for flag_short, flag_nf in flags_map.items():
        if f'{flag_short}_flag_counts' in expected_feature_list: features[f'{flag_short}_flag_counts'] = get_attr(flow, f'bidirectional_{flag_nf}_packets')
        if f'fwd_{flag_short}_flag_counts' in expected_feature_list: features[f'fwd_{flag_short}_flag_counts'] = get_attr(flow, f'src2dst_{flag_nf}_packets')
        if f'bwd_{flag_short}_flag_counts' in expected_feature_list: features[f'bwd_{flag_short}_flag_counts'] = get_attr(flow, f'dst2src_{flag_nf}_packets')
    features['packets_IAT_mean'] = get_attr(flow, 'bidirectional_mean_inter_arrival_time'); features['packet_IAT_std'] = get_attr(flow, 'bidirectional_stddev_inter_arrival_time')
    features['packet_IAT_max'] = get_attr(flow, 'bidirectional_max_inter_arrival_time'); features['packet_IAT_min'] = get_attr(flow, 'bidirectional_min_inter_arrival_time')
    features['packet_IAT_total'] = get_attr(flow, 'bidirectional_sum_inter_arrival_time'); features['fwd_packets_IAT_mean'] = get_attr(flow, 'src2dst_mean_inter_arrival_time')
    features['fwd_packets_IAT_std'] = get_attr(flow, 'src2dst_stddev_inter_arrival_time'); features['fwd_packets_IAT_max'] = get_attr(flow, 'src2dst_max_inter_arrival_time')
    features['fwd_packets_IAT_min'] = get_attr(flow, 'src2dst_min_inter_arrival_time'); features['fwd_packets_IAT_total'] = get_attr(flow, 'src2dst_sum_inter_arrival_time')
    features['bwd_packets_IAT_mean'] = get_attr(flow, 'dst2src_mean_inter_arrival_time'); features['bwd_packets_IAT_std'] = get_attr(flow, 'dst2src_stddev_inter_arrival_time')
    features['bwd_packets_IAT_max'] = get_attr(flow, 'dst2src_max_inter_arrival_time'); features['bwd_packets_IAT_min'] = get_attr(flow, 'dst2src_min_inter_arrival_time')
    features['bwd_packets_IAT_total'] = get_attr(flow, 'dst2src_sum_inter_arrival_time')
    features['subflow_fwd_packets'] = get_attr(flow, 'src2dst_packets'); features['subflow_bwd_packets'] = get_attr(flow, 'dst2src_packets')
    features['subflow_fwd_bytes'] = get_attr(flow, 'src2dst_bytes'); features['subflow_bwd_bytes'] = get_attr(flow, 'dst2src_bytes')
    if 'total_header_bytes' in expected_feature_list: features['total_header_bytes'] = get_attr(flow, 'bidirectional_header_bytes')
    if 'max_header_bytes' in expected_feature_list: features['max_header_bytes'] = get_attr(flow, 'bidirectional_max_raw_packet_size')
    if 'min_header_bytes' in expected_feature_list: features['min_header_bytes'] = get_attr(flow, 'bidirectional_min_raw_packet_size')
    if 'mean_header_bytes' in expected_feature_list: features['mean_header_bytes'] = get_attr(flow, 'bidirectional_mean_raw_packet_size')
    if 'std_header_bytes' in expected_feature_list: features['std_header_bytes'] = get_attr(flow, 'bidirectional_stddev_raw_packet_size')
    if 'fwd_max_header_bytes' in expected_feature_list: features['fwd_max_header_bytes'] = get_attr(flow, 'src2dst_max_raw_packet_size')
    if 'fwd_min_header_bytes' in expected_feature_list: features['fwd_min_header_bytes'] = get_attr(flow, 'src2dst_min_raw_packet_size')
    if 'fwd_mean_header_bytes' in expected_feature_list: features['fwd_mean_header_bytes'] = get_attr(flow, 'src2dst_mean_raw_packet_size')
    if 'fwd_std_header_bytes' in expected_feature_list: features['fwd_std_header_bytes'] = get_attr(flow, 'src2dst_stddev_raw_packet_size')
    if 'bwd_max_header_bytes' in expected_feature_list: features['bwd_max_header_bytes'] = get_attr(flow, 'dst2src_max_raw_packet_size')
    if 'bwd_min_header_bytes' in expected_feature_list: features['bwd_min_header_bytes'] = get_attr(flow, 'dst2src_min_raw_packet_size')
    if 'bwd_mean_header_bytes' in expected_feature_list: features['bwd_mean_header_bytes'] = get_attr(flow, 'dst2src_mean_raw_packet_size')
    if 'bwd_std_header_bytes' in expected_feature_list: features['bwd_std_header_bytes'] = get_attr(flow, 'dst2src_stddev_raw_packet_size')
    if 'avg_segment_size' in expected_feature_list: features['avg_segment_size'] = get_attr(flow, 'bidirectional_mean_payload_bytes')
    if 'fwd_avg_segment_size' in expected_feature_list: features['fwd_avg_segment_size'] = get_attr(flow, 'src2dst_mean_payload_bytes')
    if 'bwd_avg_segment_size' in expected_feature_list: features['bwd_avg_segment_size'] = get_attr(flow, 'dst2src_mean_payload_bytes')
    final_features_dict = {feat_name: features.get(feat_name, 0.0) for feat_name in expected_feature_list}
    return final_features_dict

# --- SHAP Explanation Helper ---
def print_shap_explanation(shap_values, model_feature_names, predicted_class_idx, model_name="Model", top_n=5):
    try:
        print(f"    SHAP Explanation for {model_name} (Predicted Class Index: {predicted_class_idx}):")
        sv_for_class = None
        if isinstance(shap_values, list): 
            if predicted_class_idx < len(shap_values): sv_for_class = shap_values[predicted_class_idx][0]
            else: print(f"      Error: Pred class idx {predicted_class_idx} out of range for SHAP list (len {len(shap_values)})."); return
        elif hasattr(shap_values, 'values'): 
            if len(shap_values.values.shape) == 3: sv_for_class = shap_values.values[0, :, predicted_class_idx]
            elif len(shap_values.values.shape) == 2: 
                if predicted_class_idx == 1: sv_for_class = shap_values.values[0, :]
                elif predicted_class_idx == 0 and hasattr(shap_values, 'base_values') and isinstance(shap_values.base_values, (list, np.ndarray)) and len(shap_values.base_values) > 1 :
                     sv_for_class = shap_values.values[0, :, 0]
                else: print(f"      SHAP values primarily for positive class; explanation for class {predicted_class_idx} needs specific SHAP output."); return
        if sv_for_class is None: print(f"      Could not determine SHAP values for class index {predicted_class_idx}."); return
        contrib_df = pd.DataFrame({'feature': model_feature_names, 'shap_value': sv_for_class})
        contrib_df['abs_shap_value'] = np.abs(contrib_df['shap_value'])
        sorted_contrib = contrib_df.sort_values(by='abs_shap_value', ascending=False)
        for i in range(min(top_n, len(sorted_contrib))):
            print(f"      - {sorted_contrib['feature'].iloc[i]}: {sorted_contrib['shap_value'].iloc[i]:.4f}")
    except Exception as e: print(f"      Error during SHAP explanation print: {e}")

# --- Main Processing Logic ---
def process_flow(flow_object):
    if not MODELS: print("Models not loaded."); return
    flow_summary = f"Flow: {flow_object.src_ip}:{flow_object.src_port} -> {flow_object.dst_ip}:{flow_object.dst_port} (Proto: {flow_object.protocol}, App: {flow_object.application_name})"
    print(f"\n--- Processing {flow_summary} ---")
    flow_features_dict = nfstream_flow_to_features_dict(flow_object, NFSTREAM_COMPATIBLE_FEATURES)
    flow_features_df = pd.DataFrame([flow_features_dict]); flow_features_df.replace([np.inf, -np.inf], np.nan, inplace=True); flow_features_df.fillna(0.0, inplace=True)

    main_pred_label = "Error/Skipped"; main_pred_idx = -1
    try:
        main_model_req_features = MODELS['main_selected_features']; main_feats_df_for_model = pd.DataFrame(0.0, index=[0], columns=main_model_req_features)
        missing_main_feats_count = 0
        for col in main_model_req_features:
            if col in flow_features_df.columns: main_feats_df_for_model.loc[0, col] = flow_features_df.loc[0, col]
            else: missing_main_feats_count +=1
        if missing_main_feats_count > 0: main_pred_label = f"Error: Missing features ({missing_main_feats_count})"
        else:
            main_feats_scaled = MODELS['main_scaler'].transform(main_feats_df_for_model); y_pred_main_raw = MODELS['main_detector'].predict(main_feats_scaled)
            y_proba_main_raw = MODELS['main_detector'].predict_proba(main_feats_scaled); y_final_refined_idx_arr = np.copy(y_pred_main_raw)
            web_probas = MODELS['web_detector'].predict_proba(main_feats_scaled)[:, 1]; web_cls_indices_in_main = []
            for cls_name_web in WEB_ATTACK_CLASSES_INF:
                try: web_cls_indices_in_main.append(MODELS['main_label_encoder'].transform([cls_name_web])[0])
                except ValueError: pass
            if web_cls_indices_in_main:
                web_detector_pos_idx = np.where(web_probas > WEB_ATTACK_REFINEMENT_THRESHOLD_INF)[0]
                for i in web_detector_pos_idx:
                    if y_pred_main_raw[i] not in web_cls_indices_in_main:
                        main_model_web_probas = y_proba_main_raw[i, web_cls_indices_in_main]
                        if np.sum(main_model_web_probas) > 1e-5:
                            best_web_cls_local_idx = np.argmax(main_model_web_probas)
                            y_final_refined_idx_arr[i] = web_cls_indices_in_main[best_web_cls_local_idx]
            main_pred_idx = y_final_refined_idx_arr[0]; main_pred_label = MODELS['main_label_encoder'].inverse_transform([main_pred_idx])[0]
            if SHAP_AVAILABLE and 'main_rf_explainer' in MODELS:
                shap_values_main = MODELS['main_rf_explainer'](main_feats_scaled)
                print_shap_explanation(shap_values_main, main_model_req_features, main_pred_idx, "Main RF Detector", SHAP_TOP_N_FEATURES)
        print(f"  Main Threat Prediction: {main_pred_label}" + (f" (Index: {main_pred_idx})" if main_pred_idx !=-1 else ""))
    except Exception as e: print(f"  Error in Main Threat Prediction: {e}"); main_pred_label = f"Error: {str(e)[:30]}"

    operator_pred_label = "Error/Skipped"; op_pred_idx = -1
    try:
        op_model_req_features = MODELS['operator_selected_features']; op_feats_df_for_model = pd.DataFrame(0.0, index=[0], columns=op_model_req_features)
        missing_op_feats_count = 0
        for col in op_model_req_features:
            if col in flow_features_df.columns: op_feats_df_for_model.loc[0, col] = flow_features_df.loc[0, col]
            else: missing_op_feats_count +=1
        if missing_op_feats_count > 0: operator_pred_label = f"Error: Missing features ({missing_op_feats_count})"
        else:
            op_feats_scaled = MODELS['operator_scaler'].transform(op_feats_df_for_model); op_pred_idx = MODELS['operator_model'].predict(op_feats_scaled)[0]
            operator_pred_label = "Bot" if op_pred_idx == 1 else "Human"
            if SHAP_AVAILABLE and 'operator_rf_explainer' in MODELS:
                shap_values_op = MODELS['operator_rf_explainer'](op_feats_scaled)
                print_shap_explanation(shap_values_op, op_model_req_features, op_pred_idx, "Operator ID RF", SHAP_TOP_N_FEATURES)
        print(f"  Operator ID Prediction: {operator_pred_label}"+ (f" (Index: {op_pred_idx})" if op_pred_idx !=-1 else ""))
    except Exception as e: print(f"  Error in Operator ID Prediction: {e}"); operator_pred_label = f"Error: {str(e)[:30]}"

    source_cat_label = "Error/Skipped"
    try:
        src_ip = flow_object.src_ip
        if src_ip:
            asn_num, asn_org = get_asn_info_for_ip_cached_inf(src_ip); is_aws, is_azure, is_gcp, is_vpn, is_tor = check_ip_membership_inf(src_ip, aws_nets_frozen_inf, azure_nets_frozen_inf, gcp_nets_frozen_inf, vpn_nets_frozen_inf, tor_nets_frozen_inf)
            is_cloud = is_aws or is_azure or is_gcp; is_proxy = is_vpn or is_tor
            source_model_features = pd.DataFrame([{'ASN': asn_num if asn_num is not None else -1, 'ASN_Org': asn_org if asn_org else 'Unknown', 'Is_Cloud': is_cloud, 'Is_Proxy': is_proxy}])
            source_model_features['ASN_Org'] = source_model_features['ASN_Org'].astype(str); source_model_features['Is_Cloud'] = source_model_features['Is_Cloud'].astype(bool); source_model_features['Is_Proxy'] = source_model_features['Is_Proxy'].astype(bool)
            source_pred_idx = MODELS['source_classifier_pipeline'].predict(source_model_features)[0]
            source_cat_label = MODELS['source_classifier_label_encoder'].inverse_transform([source_pred_idx])[0]
            print(f"  Source Category Prediction (for {src_ip}): {source_cat_label} (ASN: {asn_num} '{asn_org}', Cloud: {is_cloud}, Proxy: {is_proxy})")
        else: source_cat_label = "No Source IP"; print(f"  Source Category Prediction: No Source IP")
    except Exception as e: print(f"  Error in Source Category Prediction: {e}"); source_cat_label = f"Error: {str(e)[:30]}"

    print(f"--- Predictions for {flow_summary} ---"); print(f"  Threat Type: {main_pred_label}"); print(f"  Operator: {operator_pred_label}"); print(f"  Source Category: {source_cat_label}"); print(f"--------------------------------------")

if __name__ == "__main__":
    if not NFSTREAM_AVAILABLE: print("NFStream is not available. Exiting."); exit()
    if SHAP_AVAILABLE: print("SHAP library found. Explainability enabled.")
    
    # --- Get NFSTREAM_SOURCE from console input ---
    default_interface = '' # Or common one like 'eth0', 'en0'
    try:
        user_input_source = input(f"Enter NFStream source (e.g., interface name or path to PCAP file) [default: {default_interface}]: ")
        NFSTREAM_SOURCE = user_input_source.strip() if user_input_source.strip() else default_interface
    except EOFError: # Happens if script is piped and no input is given
        print(f"No input provided for NFStream source, using default: {default_interface}")
        NFSTREAM_SOURCE = default_interface
    print(f"Using NFStream source: {NFSTREAM_SOURCE}")
    # --- End NFSTREAM_SOURCE input ---

    if not load_all_artifacts_inference(): print("Failed to load all model artifacts. Exiting."); exit()
    
    print(f"\nStarting NFStreamer on source: {NFSTREAM_SOURCE}")
    print("Capturing traffic... Press Ctrl+C to stop.")

    streamer = NFStreamer(
        source=NFSTREAM_SOURCE, decode_tunnels=True, bpf_filter=None, promiscuous_mode=True,
        snapshot_length=1536, idle_timeout=120, active_timeout=300, accounting_mode=0,
        udps=None, n_dissections=20, statistical_analysis=True, splt_analysis=0, performance_report=0
    )
    try:
        flow_count = 0
        for flow in streamer:
            flow_count += 1;
            try: process_flow(flow)
            except Exception as e_proc: print(f"Error processing flow {flow_count}: {e_proc}"); import traceback; traceback.print_exc()
    except KeyboardInterrupt: print("\nStopping capture...")
    except RuntimeError as re: # Catch common NFStream runtime errors like interface not found
        print(f"\nNFStream RuntimeError: {re}")
        print("Please ensure the interface name is correct and you have necessary permissions (e.g., run with sudo on Linux for live capture).")
    except Exception as e: print(f"\nAn error occurred with NFStream: {e}"); import traceback; traceback.print_exc()
    finally: print("NFStreamer stopped.")