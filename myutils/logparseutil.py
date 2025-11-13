import re
import csv
from datetime import datetime
import pandas as pd
from scipy.stats import pearsonr

# Regex patterns
BLOCK_START_PATTERN = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}).*Costs per service time is not set')
JOBS_PATTERN = re.compile(r'#jobs=(\d+)')
ALGO_TIME_PATTERN = re.compile(r'took ([\d\.]+) seconds')
TIMESTAMP_PATTERN = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d{3})')
DEPOT_PATTERN = re.compile(r'Prepare for VRP: (CEPDepot\d+)')

def time_to_ms(timestamp: str) -> int:
    dt = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return dt.hour * 3600000 + dt.minute * 60000 + dt.second * 1000 + int(dt.microsecond / 1000)

def read_log_file(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return f.readlines()

def split_into_blocks(lines: list) -> list:
    blocks = []
    current_block = []
    for line in lines:
        if BLOCK_START_PATTERN.search(line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)
    return blocks

# Extract timestamp from line
def extract_timestamp(line: str) -> str | None:
    match = TIMESTAMP_PATTERN.match(line)
    return match.group(1) if match else None

# Extract vehicle blocks
def extract_vehicle_blocks(lines: list) -> list:
    blocks = []
    current_block = []
    for line in lines:
        if "Costs per service time is not set" in line:
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)
    return blocks

# Extract data from vehicle block
def extract_block_data(block: list, vehicle_id: int) -> dict | None:
    start_time = end_time = None
    jobs = algo_time = None

    for line in block:
        if not start_time:
            match = TIMESTAMP_PATTERN.match(line)
            if match:
                start_time = match.group(1)
        if match := JOBS_PATTERN.search(line):
            jobs = int(match.group(1))
        if match := ALGO_TIME_PATTERN.search(line):
            algo_time = float(match.group(1)) * 1000
        if match := TIMESTAMP_PATTERN.match(line):
            end_time = match.group(1)

    if start_time and end_time and jobs is not None and algo_time is not None:
        block_time = time_to_ms(end_time) - time_to_ms(start_time)
        return {
            'vehicle': vehicle_id,
            'number_of_jobs': jobs,
            'block_time_ms': block_time,
            'algo_time_ms': int(algo_time),
            'start_ms': time_to_ms(start_time),
            'end_ms': time_to_ms(end_time)
        }
    return None

# Parse depot blocks
def parse_depot_blocks(lines: list) -> dict:
    depot_blocks = {}
    current_depot = None
    current_lines = []

    for line in lines:
        if "Prepare for VRP: CEPDepot" in line:
            if current_depot and current_lines:
                depot_blocks[current_depot] = current_lines
            current_depot = DEPOT_PATTERN.search(line).group(1)
            current_lines = [line]
        elif current_depot:
            current_lines.append(line)

    if current_depot and current_lines:
        depot_blocks[current_depot] = current_lines

    return depot_blocks

# CEPDepot summary table
def build_cep_summary(depot_blocks: dict) -> pd.DataFrame:
    summary = []
    vehicle_id = 1

    for depot, lines in depot_blocks.items():
        vehicle_blocks = extract_vehicle_blocks(lines)
        total_jobs = 0
        start_time = end_time = None

        for block in vehicle_blocks:
            data = extract_block_data(block, vehicle_id)
            vehicle_id += 1
            if data:
                total_jobs += data['number_of_jobs']
                if start_time is None or data['start_ms'] < start_time:
                    start_time = data['start_ms']
                if end_time is None or data['end_ms'] > end_time:
                    end_time = data['end_ms']

        if start_time and end_time:
            summary.append({
                'CEPDepot': depot,
                'total_jobs': total_jobs,
                'operation_time_ms': end_time - start_time
            })

    return pd.DataFrame(summary)

def ms_to_hh_mm_ss(ms: int) -> str:
    """Convert milliseconds to 'HH:MM:SS' string (zero-padded)."""
    s_total = ms // 1000
    hours = s_total // 3600
    minutes = (s_total % 3600) // 60
    seconds = s_total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Simulation phase timing table
def build_phase_timing(lines: list) -> pd.DataFrame:
    first_ts = extract_timestamp(lines[0])
    first_vrp_ts = next((extract_timestamp(line) for line in lines if "Prepare for VRP" in line), None)
    iteration_start_ts = next((extract_timestamp(line) for line in lines if "all ControlerIterationStartsListeners called" in line), None)
    shutdown_start_ts = next((extract_timestamp(line) for line in lines if "start shutdown" in line), None)
    final_ts = extract_timestamp(lines[-1])

    times = {
        'Preprocessing': time_to_ms(first_vrp_ts) - time_to_ms(first_ts),
        'CEPDepot Ops': time_to_ms(iteration_start_ts) - time_to_ms(first_vrp_ts),
        'Iteration': time_to_ms(shutdown_start_ts) - time_to_ms(iteration_start_ts),
        'Post-process': time_to_ms(final_ts) - time_to_ms(shutdown_start_ts),
        'Total': time_to_ms(final_ts) - time_to_ms(first_ts)
    }

    # Build DataFrame
    df = pd.DataFrame([{'Phase': k, 'Duration_ms': v} for k, v in times.items()])

    # Add formatted duration column and share of total time (percentage)
    total_ms = int(df.loc[df['Phase'] == 'Total', 'Duration_ms'].iloc[0]) if 'Total' in df['Phase'].values else df['Duration_ms'].sum()
    df['hh_mm_ss'] = df['Duration_ms'].apply(ms_to_hh_mm_ss)
    df['share_of_total_time'] = (df['Duration_ms'] / total_ms * 100).round(2)

    return df

# Run everything
def analyze_log(filepath: str):
    lines = read_log_file(filepath)
    depot_blocks = parse_depot_blocks(lines)

    cep_summary_df = build_cep_summary(depot_blocks)
    phase_timing_df = build_phase_timing(lines)

    return cep_summary_df, phase_timing_df

def write_csv(data: list, output_file: str):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['vehicle', 'number_of_jobs', 'block_time_ms', 'algo_time_ms'])
        writer.writeheader()
        writer.writerows(data)