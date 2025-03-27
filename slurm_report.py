#!/usr/bin/env python3
"""
Enhanced Slurm Usage Report Generator
-------------------------------------
This script generates a comprehensive HTML report of Slurm cluster usage with interactive
JavaScript visualizations. It automatically runs Slurm commands and processes the output
to create an informative and visually appealing report.

Features:
- Interactive charts using Chart.js
- QOS settings table
- User QOS assignments table
- Detailed executive summary with key metrics
- Customizable date ranges and filtering options

To run (as root):
```
python3 slurm_report.py --start-date 2025-01-01
```
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import io
from jinja2 import Template
import os
import re

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate an enhanced Slurm usage report')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format (default: 30 days ago)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--user', type=str, default=None,
                        help='Filter by specific user (default: all users)')
    parser.add_argument('--partition', type=str, default=None,
                        help='Filter by specific partition (default: all partitions)')
    parser.add_argument('--output', type=str, default='slurm_report.html',
                        help='Output HTML file name (default: slurm_report.html)')
    parser.add_argument('--exclude-qos', type=str, default=None,
                        help='Comma-separated list of QOS values to exclude from the report')
    
    return parser.parse_args()

def run_command(cmd):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error message: {e.stderr}")
        return None

def get_qos_settings():
    """Get QOS settings from the Slurm database."""
    cmd = ["sacctmgr", "show", "qos", "-s", "-P", "format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU"]
    output = run_command(cmd)
    
    if not output:
        return pd.DataFrame()
    
    # Parse the output
    df = pd.read_csv(io.StringIO(output), sep='|')
    
    # Clean up column names (remove trailing spaces)
    df.columns = [col.strip() for col in df.columns]

    # Filter out normal
    df = df[df['Name'] != 'normal']

    # Sort rows by name
    df = df.sort_values(by=['Name'])
    
    return df

def get_user_qos():
    """Get user QOS assignments from the Slurm database."""
    cmd = ["sacctmgr", "show", "user", "-s", "-P", "format=User,QOS"]
    output = run_command(cmd)
    
    if not output:
        return pd.DataFrame()
    
    # Parse the output
    df = pd.read_csv(io.StringIO(output), sep='|')
    
    # Clean up column names (remove trailing spaces)
    df.columns = [col.strip() for col in df.columns]
    
    # Expand comma-separated QOS values into separate rows
    if 'QOS' in df.columns:
        result = []
        for _, row in df.iterrows():
            user = row['User']
            qos_list = list(sorted(row['QOS'].split(',')))
            qos_list = ', '.join(qos_list)
            result.append({'User': user, 'QOS': qos_list})
        if result:
            df = pd.DataFrame(result)
    
    return df

def run_sacct_command(start_date=None, end_date=None, user=None, partition=None):
    """Run the sacct command and return its output."""
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d-00:00:00')
    else:
        start_date = f"{start_date}-00:00:00"
    
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d-23:59:59')
    else:
        end_date = f"{end_date}-23:59:59"
    
    # Build the sacct command with more detailed information
    cmd = ["sacct", "--parsable", "--starttime", f"{start_date}", "--endtime", f"{end_date}",
           "-ojobid,start,end,state,nodelist,user,qos,partition,timelimit,submit,elapsed,allocTRES,reqTRES,reqcpus,reqmem,ncpus,nnodes,exitcode,reason"]
    
    # Add user filter if specified
    if user:
        cmd.extend(["-u", user])
    
    # Add partition filter if specified
    if partition:
        cmd.extend(["-r", partition])
    
    # Run the command
    return run_command(cmd)

def process_data(sacct_output):
    """Process the sacct output into a pandas DataFrame."""
    if not sacct_output:
        return pd.DataFrame()
    
    # Load data from string
    df = pd.read_csv(io.StringIO(sacct_output), sep='|')
    
    # Clean up column names (remove trailing spaces)
    df.columns = [col.strip() for col in df.columns]
    
    # Convert time columns to datetime
    time_cols = ['Start', 'End', 'Submit']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate queue time in seconds
    if 'Start' in df.columns and 'Submit' in df.columns:
        df['QueueTime'] = (df['Start'] - df['Submit']).dt.total_seconds()
    
    # Calculate job run time in seconds
    if 'End' in df.columns and 'Start' in df.columns:
        df['ElapsedTimeSec'] = (df['End'] - df['Start']).dt.total_seconds()
    
    # Extract CPU and memory requests
    if 'ReqCPUS' in df.columns:
        df['ReqCPUS'] = pd.to_numeric(df['ReqCPUS'], errors='coerce').fillna(0).astype(int)
    
    if 'ReqMem' in df.columns:
        # Extract memory values (handle formats like '16G' or '1000M')
        df['MemoryRequestGB'] = df['ReqMem'].apply(extract_memory_gb)
    
    # Extract GPU requests from ReqTRES
    if 'ReqTRES' in df.columns:
        df['ReqGPUs'] = df['ReqTRES'].apply(
            lambda x: extract_gpu_count(x) if isinstance(x, str) else 0
        )
    
    # Extract allocated GPUs from AllocTRES
    if 'AllocTRES' in df.columns:
        df['AllocGPUs'] = df['AllocTRES'].apply(
            lambda x: extract_gpu_count(x) if isinstance(x, str) else 0
        )
    
    # Handle Timelimit conversion
    if 'Timelimit' in df.columns:
        df['TimelimitSec'] = df['Timelimit'].apply(convert_timelimit_to_seconds)
    
    # Convert Elapsed to seconds if needed
    if 'Elapsed' in df.columns:
        df['ElapsedCalculated'] = df['Elapsed'].apply(
            lambda x: convert_elapsed_to_seconds(x) if isinstance(x, str) else 0
        )
    
    # Add date columns for grouping
    if 'Submit' in df.columns:
        df['SubmitDate'] = df['Submit'].dt.date
        df['SubmitHour'] = df['Submit'].dt.hour
        df['SubmitDayOfWeek'] = df['Submit'].dt.dayofweek
        df['SubmitMonth'] = df['Submit'].dt.month
    
    # Clean up job states (handle cancellations)
    if 'State' in df.columns:
        df['CleanState'] = df['State'].apply(
            lambda x: extract_clean_state(x)
        )
    
    # Calculate GPU and CPU time used
    if 'ReqGPUs' in df.columns and 'ElapsedTimeSec' in df.columns:
        df['GPUTimeUsed'] = df['ReqGPUs'] * df['ElapsedTimeSec'] / 3600  # in hours
    
    if 'NCPUS' in df.columns and 'ElapsedTimeSec' in df.columns:
        df['CPUTimeUsed'] = df['NCPUS'] * df['ElapsedTimeSec'] / 3600  # in CPU-hours
    
    return df

def extract_memory_gb(mem_str):
    """Extract memory in GB from Slurm memory string format."""
    if not isinstance(mem_str, str):
        return 0
    
    # Match patterns like '16G' or '1000M'
    match_g = re.search(r'(\d+)G', mem_str)
    match_m = re.search(r'(\d+)M', mem_str)
    
    if match_g:
        return float(match_g.group(1))
    elif match_m:
        return float(match_m.group(1)) / 1024
    else:
        return 0

def extract_gpu_count(tres_str):
    """Extract GPU count from TRES string."""
    if not isinstance(tres_str, str):
        return 0
    
    # Match pattern for GPU counts in TRES
    match = re.search(r'gres/gpu=(\d+)', tres_str)
    if match:
        return int(match.group(1))
    return 0

def extract_clean_state(state_str):
    """Extract a clean job state from potentially complex state string."""
    if not isinstance(state_str, str):
        return "UNKNOWN"
    
    if 'CANCELLED by' in state_str:
        return "CANCELLED"
    elif 'TIMEOUT' in state_str:
        return "TIMEOUT"
    elif 'FAILED' in state_str:
        return "FAILED"
    elif 'COMPLETED' in state_str:
        return "COMPLETED"
    else:
        return state_str

def convert_timelimit_to_seconds(timelimit):
    """Convert a Timelimit string to seconds."""
    if not isinstance(timelimit, str) or timelimit == 'UNLIMITED':
        return np.nan
    
    try:
        # Splitting days from time if present
        if '-' in timelimit:
            days, time = timelimit.split('-')
            days = int(days)
        else:
            days = 0
            time = timelimit
        
        # Handle different time formats
        if time.count(':') == 2:
            hours, minutes, seconds = map(int, time.split(':'))
        elif time.count(':') == 1:
            hours, minutes = map(int, time.split(':'))
            seconds = 0
        else:
            # If only minutes are specified
            hours, minutes, seconds = 0, int(time), 0
        
        # Convert everything to seconds
        total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
        return total_seconds
    except Exception:
        return np.nan

def convert_elapsed_to_seconds(elapsed):
    """Convert Elapsed time string to seconds."""
    if not isinstance(elapsed, str):
        return 0
    
    try:
        # Handle formats like '1-12:30:45' (days-hours:minutes:seconds)
        if '-' in elapsed:
            days, time = elapsed.split('-')
            days = int(days)
        else:
            days = 0
            time = elapsed
        
        # Handle time part
        parts = time.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours, minutes = map(int, parts)
            seconds = 0
        else:
            # If only one value, assume it's minutes
            hours, minutes, seconds = 0, int(parts[0]), 0
        
        # Calculate total seconds
        return (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
    except Exception:
        return 0

def prepare_chart_data(df, exclude_qos=None):
    """Prepare data for interactive charts."""
    chart_data = {}
    
    # Filter out excluded QOS values if specified
    if exclude_qos:
        exclude_list = [qos.strip() for qos in exclude_qos.split(',')]
        df = df[~df['QOS'].isin(exclude_list)]
    
    # Only proceed if we have data
    if df.empty:
        return {}
    
    # 1. Job state distribution
    if 'CleanState' in df.columns:
        state_counts = df['CleanState'].value_counts()
        chart_data['job_states'] = {
            'labels': state_counts.index.tolist(),
            'data': state_counts.values.tolist()
        }
    
    # 2. QOS usage distribution
    if 'QOS' in df.columns:
        qos_counts = df['QOS'].value_counts()
        chart_data['qos_usage'] = {
            'labels': qos_counts.index.tolist(),
            'data': qos_counts.values.tolist()
        }
    
    # 3. Jobs per user (top 15)
    if 'User' in df.columns:
        user_jobs = df['User'].value_counts().head(15)
        chart_data['jobs_per_user'] = {
            'labels': user_jobs.index.tolist(),
            'data': user_jobs.values.tolist()
        }
    
    # 4. Job submissions over time
    if 'SubmitDate' in df.columns:
        daily_jobs = df.groupby('SubmitDate').size()
        
        # Create date strings for x-axis
        dates = [date.strftime('%Y-%m-%d') for date in daily_jobs.index]
        
        chart_data['jobs_over_time'] = {
            'labels': dates,
            'data': daily_jobs.values.tolist()
        }
    
    # 5. Job submissions by hour of day
    if 'SubmitHour' in df.columns:
        hourly_jobs = df.groupby('SubmitHour').size().reindex(range(24), fill_value=0)
        chart_data['jobs_by_hour'] = {
            'labels': list(range(24)),
            'data': hourly_jobs.values.tolist()
        }
    
    # 6. Job submissions by day of week
    if 'SubmitDayOfWeek' in df.columns:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_jobs = df.groupby('SubmitDayOfWeek').size().reindex(range(7), fill_value=0)
        chart_data['jobs_by_weekday'] = {
            'labels': day_names,
            'data': weekday_jobs.values.tolist()
        }
    
    # 7. Average queue time per QOS
    if 'QOS' in df.columns and 'QueueTime' in df.columns:
        avg_queue_time = df.groupby('QOS')['QueueTime'].mean() / 60  # in minutes
        chart_data['queue_time_qos'] = {
            'labels': avg_queue_time.index.tolist(),
            'data': avg_queue_time.values.tolist()
        }
    
    # 8. GPU usage per user (top 15)
    if 'User' in df.columns and 'GPUTimeUsed' in df.columns:
        gpu_df = df[df['GPUTimeUsed'] > 0]
        if not gpu_df.empty:
            gpu_usage = gpu_df.groupby('User')['GPUTimeUsed'].sum().sort_values(ascending=False).head(15)
            chart_data['gpu_usage'] = {
                'labels': gpu_usage.index.tolist(),
                'data': gpu_usage.values.tolist()
            }
    
    # 9. Job runtime distribution per QOS (boxplot data)
    if 'QOS' in df.columns and 'ElapsedTimeSec' in df.columns:
        # Filter out outliers and irrelevant QOS values
        filtered_df = df.copy()
        filtered_df['ElapsedTimeHours'] = filtered_df['ElapsedTimeSec'] / 3600
        
        # Prepare boxplot data for each QOS
        boxplot_data = {}
        for qos in filtered_df['QOS'].unique():
            qos_data = filtered_df[filtered_df['QOS'] == qos]['ElapsedTimeHours'].dropna()
            if not qos_data.empty:
                boxplot_data[qos] = {
                    'min': qos_data.min(),
                    'q1': qos_data.quantile(0.25),
                    'mean': qos_data.mean(),
                    'median': qos_data.median(),
                    'std': qos_data.std(ddof=0),
                    'q3': qos_data.quantile(0.75),
                    'max': qos_data.quantile(0.95),  # Using 95th percentile instead of max to avoid extreme outliers
                    'count': len(qos_data)
                }
        
        chart_data['job_runtime_boxplot'] = boxplot_data
    
    # 10. Memory usage per QOS
    if 'QOS' in df.columns and 'MemoryRequestGB' in df.columns:
        avg_memory = df.groupby('QOS')['MemoryRequestGB'].mean()
        chart_data['memory_per_qos'] = {
            'labels': avg_memory.index.tolist(),
            'data': avg_memory.values.tolist()
        }
    
    # 11. CPU usage per user (top 15)
    if 'User' in df.columns and 'CPUTimeUsed' in df.columns:
        cpu_usage = df.groupby('User')['CPUTimeUsed'].sum().sort_values(ascending=False).head(15)
        chart_data['cpu_usage'] = {
            'labels': cpu_usage.index.tolist(),
            'data': cpu_usage.values.tolist()
        }
    
    # 12. Job completion rate by user (for users with > 10 jobs)
    if 'User' in df.columns and 'CleanState' in df.columns:
        # Count total jobs per user
        user_job_counts = df['User'].value_counts()
        users_with_many_jobs = user_job_counts[user_job_counts >= 10].index
        
        # Calculate completion rates
        completion_rates = {}
        for user in users_with_many_jobs:
            user_df = df[df['User'] == user]
            total = len(user_df)
            completed = len(user_df[user_df['CleanState'] == 'COMPLETED'])
            completion_rates[user] = (completed / total) * 100 if total > 0 else 0
        
        # Sort by completion rate
        sorted_rates = sorted(completion_rates.items(), key=lambda x: x[1], reverse=True)
        
        chart_data['completion_rate_by_user'] = {
            'labels': [item[0] for item in sorted_rates],
            'data': [item[1] for item in sorted_rates]
        }
    
    return chart_data

def calculate_statistics(df):
    """Calculate comprehensive statistics from the data."""
    stats = {}
    
    if df.empty:
        return stats
    
    # Basic job stats
    stats['total_jobs'] = len(df)
    
    if 'CleanState' in df.columns:
        for state in df['CleanState'].unique():
            count = len(df[df['CleanState'] == state])
            stats[f'jobs_{state.lower()}'] = count
        
        completed_jobs = df[df['CleanState'] == 'COMPLETED']
        stats['completed_jobs'] = len(completed_jobs)
        stats['completion_rate'] = len(completed_jobs) / len(df) * 100 if len(df) > 0 else 0
    
    # Time-based stats
    if 'QueueTime' in df.columns:
        queue_times = df['QueueTime'].dropna()
        stats['avg_queue_time_min'] = queue_times.mean() / 60 if not queue_times.empty else 0
        stats['median_queue_time_min'] = queue_times.median() / 60 if not queue_times.empty else 0
        stats['max_queue_time_hours'] = queue_times.max() / 3600 if not queue_times.empty else 0
        stats['p95_queue_time_hours'] = queue_times.quantile(0.95) / 3600 if not queue_times.empty else 0
    
    if 'ElapsedTimeSec' in df.columns:
        run_times = df['ElapsedTimeSec'].dropna()
        stats['avg_runtime_hours'] = run_times.mean() / 3600 if not run_times.empty else 0
        stats['median_runtime_hours'] = run_times.median() / 3600 if not run_times.empty else 0
        stats['total_runtime_days'] = run_times.sum() / (3600 * 24) if not run_times.empty else 0
    
    # Resource stats
    if 'ReqCPUS' in df.columns:
        stats['total_cpu_requests'] = df['ReqCPUS'].sum()
        stats['avg_cpus_per_job'] = df['ReqCPUS'].mean() if not df['ReqCPUS'].empty else 0
    
    if 'CPUTimeUsed' in df.columns:
        stats['total_cpu_hours'] = df['CPUTimeUsed'].sum() if 'CPUTimeUsed' in df else 0
    
    if 'MemoryRequestGB' in df.columns:
        stats['avg_memory_request_gb'] = df['MemoryRequestGB'].mean() if not df['MemoryRequestGB'].empty else 0
        stats['total_memory_request_gb'] = df['MemoryRequestGB'].sum() if not df['MemoryRequestGB'].empty else 0
    
    # User stats
    if 'User' in df.columns:
        stats['unique_users'] = df['User'].nunique()
        
        # Average jobs per user
        stats['avg_jobs_per_user'] = len(df) / df['User'].nunique() if df['User'].nunique() > 0 else 0
        
        # Get top user by job count
        top_user = df['User'].value_counts().idxmax() if not df['User'].empty else 'None'
        stats['top_user'] = top_user
        stats['top_user_jobs'] = df['User'].value_counts().max() if not df['User'].empty else 0
        
        # Get top user by GPU usage
        if 'GPUTimeUsed' in df.columns:
            gpu_usage = df.groupby('User')['GPUTimeUsed'].sum()
            if not gpu_usage.empty:
                stats['top_gpu_user'] = gpu_usage.idxmax()
                stats['top_gpu_user_hours'] = gpu_usage.max()
    
    # GPU stats
    if 'ReqGPUs' in df.columns:
        stats['total_gpu_requests'] = df['ReqGPUs'].sum()
        stats['avg_gpus_per_job'] = df['ReqGPUs'].mean() if df['ReqGPUs'].sum() > 0 else 0
        
    if 'GPUTimeUsed' in df.columns:
        gpu_times = df['GPUTimeUsed'].dropna()
        stats['total_gpu_hours'] = gpu_times.sum() if not gpu_times.empty else 0
    
    # Efficiency stats
    if 'TimelimitSec' in df.columns and 'ElapsedTimeSec' in df.columns:
        # Calculate how effectively time limits are being used
        efficiency_data = df[df['TimelimitSec'].notna() & df['ElapsedTimeSec'].notna()]
        if not df[df['TimelimitSec'].notna() & df['ElapsedTimeSec'].notna()].empty:
            # Create an explicit copy
            efficiency_data = df[df['TimelimitSec'].notna() & df['ElapsedTimeSec'].notna()].copy()
            # Now you can modify it
            efficiency_data['TimeUsageRatio'] = efficiency_data['ElapsedTimeSec'] / efficiency_data['TimelimitSec']
            stats['avg_time_usage_ratio'] = efficiency_data['TimeUsageRatio'].mean() * 100  # as percentage
    
    # Partition stats
    if 'Partition' in df.columns:
        stats['partitions_used'] = df['Partition'].nunique()
        stats['top_partition'] = df['Partition'].value_counts().idxmax() if not df['Partition'].empty else 'None'
    
    # QOS stats
    if 'QOS' in df.columns:
        stats['qos_used'] = df['QOS'].nunique()
        stats['top_qos'] = df['QOS'].value_counts().idxmax() if not df['QOS'].empty else 'None'
    
    # Node stats
    if 'NodeList' in df.columns:
        # Count unique nodes used
        all_nodes = set()
        for nodelist in df['NodeList'].dropna():
            if isinstance(nodelist, str):
                # Handle comma-separated nodelists and ranges (node[1-3])
                parts = nodelist.split(',')
                for part in parts:
                    if '[' in part and ']' in part:
                        # Handle node ranges like "node[1-3,5,7-9]"
                        node_base = part.split('[')[0]
                        ranges = part.split('[')[1].split(']')[0].split(',')
                        for r in ranges:
                            if '-' in r:
                                start, end = map(int, r.split('-'))
                                for i in range(start, end + 1):
                                    all_nodes.add(f"{node_base}{i}")
                            else:
                                all_nodes.add(f"{node_base}{r}")
                    else:
                        all_nodes.add(part)
        
        stats['unique_nodes_used'] = len(all_nodes)
    
    # Date range
    if 'Submit' in df.columns:
        submit_dates = df['Submit'].dropna()
        if not submit_dates.empty:
            stats['start_date'] = submit_dates.min().strftime('%Y-%m-%d')
            stats['end_date'] = submit_dates.max().strftime('%Y-%m-%d')
            stats['date_range_days'] = (submit_dates.max() - submit_dates.min()).days + 1
    
    # Load patterns
    if 'SubmitHour' in df.columns:
        hourly_jobs = df.groupby('SubmitHour').size()
        if not hourly_jobs.empty:
            stats['peak_hour'] = hourly_jobs.idxmax()
            stats['peak_hour_jobs'] = hourly_jobs.max()
    
    if 'SubmitDayOfWeek' in df.columns:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_jobs = df.groupby('SubmitDayOfWeek').size()
        if not weekday_jobs.empty:
            stats['peak_day'] = day_names[weekday_jobs.idxmax()]
            stats['peak_day_jobs'] = weekday_jobs.max()
    
    return stats

def generate_html_report(chart_data, stats, qos_settings, user_qos, args):
    """Generate an HTML report with interactive charts using Chart.js."""
    template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SARDINE Cluster Usage Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment@1.0.1/dist/chartjs-adapter-moment.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.0/dist/chartjs-plugin-zoom.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-error-bars@4.4.4/build/index.umd.min.js"></script>
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --success-color: #2ecc71;
                --warning-color: #f39c12;
                --danger-color: #e74c3c;
                --info-color: #1abc9c;
                --light-color: #ecf0f1;
                --dark-color: #34495e;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding-top: 2rem;
                padding-bottom: 2rem;
                background-color: var(--light-color);
                color: var(--primary-color);
                line-height: 1.6;
            }
            
            .report-header {
                padding: 2.5rem 1.5rem;
                text-align: center;
                background: linear-gradient(135deg, var(--primary-color), var(--dark-color));
                color: white;
                margin-bottom: 2.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .report-section {
                margin-bottom: 3rem;
                background-color: white;
                padding: 2rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            .report-section h2 {
                color: var(--primary-color);
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid var(--light-color);
            }
            
            .stat-card {
                background-color: var(--light-color);
                padding: 1.25rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
                border-left: 5px solid var(--secondary-color);
                transition: transform 0.2s ease-in-out;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
            }
            
            .stat-card.success {
                border-left-color: var(--success-color);
            }
            
            .stat-card.warning {
                border-left-color: var(--warning-color);
            }
            
            .stat-card.danger {
                border-left-color: var(--danger-color);
            }
            
            .stat-card.info {
                border-left-color: var(--info-color);
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: bold;
                color: var(--secondary-color);
                margin-bottom: 0.25rem;
            }
            
            .stat-card.success .stat-value {
                color: var(--success-color);
            }
            
            .stat-card.warning .stat-value {
                color: var(--warning-color);
            }
            
            .stat-card.danger .stat-value {
                color: var(--danger-color);
            }
            
            .stat-card.info .stat-value {
                color: var(--info-color);
            }
            
            .stat-label {
                color: var(--primary-color);
                font-weight: 500;
                font-size: 1rem;
            }
            
            .chart-container {
                position: relative;
                height: 400px;
                margin-bottom: 2rem;
            }
            
            .table-responsive {
                border-radius: 0.5rem;
                overflow: hidden;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
            }
            
            .table {
                margin-bottom: 0;
            }
            
            .table th {
                background-color: var(--primary-color);
                color: white;
                font-weight: 500;
            }
            
            .qos-table th {
                background-color: var(--secondary-color);
            }
            
            .user-table th {
                background-color: var(--info-color);
            }
            
            footer {
                text-align: center;
                padding: 2rem 1rem;
                margin-top: 3rem;
                color: var(--primary-color);
                font-size: 0.9rem;
            }
            
            .nav-tabs .nav-link {
                color: var(--primary-color);
                border: none;
                font-weight: 500;
                padding: 0.75rem 1rem;
            }
            
            .nav-tabs .nav-link.active {
                color: var(--secondary-color);
                background-color: transparent;
                border-bottom: 3px solid var(--secondary-color);
            }
            
            .tab-pane {
                padding: 1.5rem 0;
            }
            .pdf-button {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s;
                margin: 0;
            }

            .pdf-button:hover {
                background-color: #34495e;
            }
            @media (max-width: 768px) {
                .stat-card {
                    margin-bottom: 1rem;
                }
                
                .chart-container {
                    height: 300px;
                }
            }
            @media print {
                body {
                    background-color: white !important;
                }
                .report-section {
                    break-inside: avoid;
                    page-break-inside: avoid;
                    background-color: white !important;
                    box-shadow: none !important;
                }
                .pdf-button, .nav-tabs button:not(.active) {
                    display: none !important;
                }
                .chart-container {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                .top-button-container, footer {
                    display: none !important;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <h1 class="display-4">SARDINE Cluster Usage Report 🐟</h1>
                <p class="lead">
                    {% if stats.start_date and stats.end_date %}
                        {{ stats.start_date }} to {{ stats.end_date }} ({{ stats.date_range_days }} days)
                    {% else %}
                        Report generated on {{ current_date }}
                    {% endif %}
                </p>
                {% if args.user %}
                    <p class="lead">User: {{ args.user }}</p>
                {% endif %}
                {% if args.partition %}
                    <p class="lead">Partition: {{ args.partition }}</p>
                {% endif %}
                <p class="lead">
                    <button class="pdf-button" id="exportPdfTop">
                        <i class="fas fa-file-pdf"></i> Export to PDF
                    </button>
                </p>
            </div>

            
            <!-- Executive Summary -->
            <div class="report-section">
                <h2>Executive Summary</h2>
                <div class="row">
                    <div class="col-md-4">
                        <div class="stat-card">
                            <p class="stat-label"><i class="fas fa-tasks me-2"></i>Total Jobs</p>
                            <p class="stat-value">{{ stats.total_jobs|default('N/A')|int }}</p>
                            <small>During the reporting period</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card success">
                            <p class="stat-label"><i class="fas fa-check-circle me-2"></i>Completion Rate</p>
                            <p class="stat-value">{{ "%.1f%%"|format(stats.completion_rate|default(0)) }}</p>
                            <small>Jobs successfully completed</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card info">
                            <p class="stat-label"><i class="fas fa-users me-2"></i>Unique Users</p>
                            <p class="stat-value">{{ stats.unique_users|default('N/A')|int }}</p>
                            <small>Active on the cluster</small>
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="stat-card warning">
                            <p class="stat-label"><i class="fas fa-hourglass-half me-2"></i>Avg Queue Time</p>
                            <p class="stat-value">{{ "%.1f min"|format(stats.avg_queue_time_min|default(0)) }}</p>
                            <small>Average job wait time</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card success">
                            <p class="stat-label"><i class="fas fa-microchip me-2"></i>Total GPU Hours</p>
                            <p class="stat-value">{{ "%.1f"|format(stats.total_gpu_hours|default(0)) }}</p>
                            <small>GPU computation hours</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card">
                            <p class="stat-label"><i class="fas fa-server me-2"></i>Total CPU Hours</p>
                            <p class="stat-value">{{ "%.1f"|format(stats.total_cpu_hours|default(0)) }}</p>
                            <small>CPU computation hours</small>
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="stat-card info">
                            <p class="stat-label"><i class="fas fa-tachometer-alt me-2"></i>Resource Efficiency</p>
                            <p class="stat-value">{{ "%.1f%%"|format(stats.avg_time_usage_ratio|default(0)) }}</p>
                            <small>Avg. % of allocated time used</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card danger">
                            <p class="stat-label"><i class="fas fa-exclamation-triangle me-2"></i>Failed Jobs</p>
                            <p class="stat-value">{{ stats.jobs_failed|default(0)|int }}</p>
                            <small>Jobs with errors or timeouts</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card warning">
                            <p class="stat-label"><i class="fas fa-calendar-day me-2"></i>Peak Usage Day</p>
                            <p class="stat-value">{{ stats.peak_day|default('N/A') }}</p>
                            <small>{{ stats.peak_day_jobs|default(0)|int }} jobs</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- QOS Settings -->
            <div class="report-section">
                <h2>QOS Settings</h2>
                <div class="table-responsive qos-table">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                {% for col in qos_settings.columns %}
                                <th>{{ col }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in qos_settings.iterrows() %}
                            <tr>
                                {% for col in qos_settings.columns %}
                                <td>{{ row[col] }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- User QOS Assignments -->
            <div class="report-section">
                <h2>User QOS Assignments</h2>
                <div class="table-responsive user-table">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>User</th>
                                <th>QOS</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in user_qos.iterrows() %}
                            <tr>
                                <td>{{ row['User'] }}</td>
                                <td>{{ row['QOS'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Job Statistics -->
            <div class="report-section">
                <h2>Job Statistics</h2>
                
                <ul class="nav nav-tabs mb-4" id="jobsTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="state-tab" data-bs-toggle="tab" data-bs-target="#state" type="button" role="tab" aria-controls="state" aria-selected="true">
                            Job States
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="timeline-tab" data-bs-toggle="tab" data-bs-target="#timeline" type="button" role="tab" aria-controls="timeline" aria-selected="false">
                            Timeline
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="users-tab" data-bs-toggle="tab" data-bs-target="#users" type="button" role="tab" aria-controls="users" aria-selected="false">
                            Users
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="patterns-tab" data-bs-toggle="tab" data-bs-target="#patterns" type="button" role="tab" aria-controls="patterns" aria-selected="false">
                            Patterns
                        </button>
                    </li>
                </ul>
                
                <div class="tab-content" id="jobsTabContent">
                    <div class="tab-pane fade show active" id="state" role="tabpanel" aria-labelledby="state-tab">
                        <div class="chart-container">
                            <canvas id="jobStatesChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="timeline" role="tabpanel" aria-labelledby="timeline-tab">
                        <div class="chart-container">
                            <canvas id="jobsOverTimeChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="users" role="tabpanel" aria-labelledby="users-tab">
                        <div class="chart-container">
                            <canvas id="jobsPerUserChart"></canvas>
                        </div>
                        <div class="chart-container mt-4">
                            <canvas id="completionRateChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="patterns" role="tabpanel" aria-labelledby="patterns-tab">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="jobsByHourChart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="jobsByWeekdayChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Resource Utilization -->
            <div class="report-section">
                <h2>Resource Utilization</h2>
                
                <ul class="nav nav-tabs mb-4" id="resourcesTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="qos-tab" data-bs-toggle="tab" data-bs-target="#qos" type="button" role="tab" aria-controls="qos" aria-selected="true">
                            QOS Usage
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="gpu-tab" data-bs-toggle="tab" data-bs-target="#gpu" type="button" role="tab" aria-controls="gpu" aria-selected="false">
                            GPU Usage
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="cpu-tab" data-bs-toggle="tab" data-bs-target="#cpu" type="button" role="tab" aria-controls="cpu" aria-selected="false">
                            CPU Usage
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="runtime-tab" data-bs-toggle="tab" data-bs-target="#runtime" type="button" role="tab" aria-controls="runtime" aria-selected="false">
                            Job Runtime
                        </button>
                    </li>
                </ul>
                
                <div class="tab-content" id="resourcesTabContent">
                    <div class="tab-pane fade show active" id="qos" role="tabpanel" aria-labelledby="qos-tab">
                        <div class="chart-container">
                            <canvas id="qosUsageChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="gpu" role="tabpanel" aria-labelledby="gpu-tab">
                        <div class="chart-container">
                            <canvas id="gpuUsageChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="cpu" role="tabpanel" aria-labelledby="cpu-tab">
                        <div class="chart-container">
                            <canvas id="cpuUsageChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="runtime" role="tabpanel" aria-labelledby="runtime-tab">
                        <div class="chart-container">
                            <canvas id="jobRuntimeChart"></canvas>
                        </div>
                        <div class="chart-container mt-4">
                            <canvas id="memoryUsageChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Queue Performance -->
            <div class="report-section">
                <h2>Queue Performance</h2>
                <div class="chart-container">
                    <canvas id="queueTimeChart"></canvas>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="stat-card warning">
                            <p class="stat-label">Maximum Queue Time</p>
                            <p class="stat-value">{{ "%.1f hrs"|format(stats.max_queue_time_hours|default(0)) }}</p>
                            <small>Longest wait time for a job</small>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="stat-card info">
                            <p class="stat-label">95th Percentile Queue Time</p>
                            <p class="stat-value">{{ "%.1f hrs"|format(stats.p95_queue_time_hours|default(0)) }}</p>
                            <small>95% of jobs waited less than this</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Additional Details -->
            <div class="report-section">
                <h2>Additional Details</h2>
                <div class="row">
                    <div class="col-md-6">
                        <h4 class="mb-3">User Statistics</h4>
                        <table class="table table-striped table-hover">
                            <tbody>
                                <tr>
                                    <td class="fw-bold">Top User (Jobs)</td>
                                    <td>{{ stats.top_user|default('N/A') }} ({{ stats.top_user_jobs|default(0)|int }} jobs)</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Top GPU User</td>
                                    <td>{{ stats.top_gpu_user|default('N/A') }} ({{ "%.1f"|format(stats.top_gpu_user_hours|default(0)) }} hours)</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Average Jobs per User</td>
                                    <td>{{ "%.1f"|format(stats.avg_jobs_per_user|default(0)) }}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h4 class="mb-3">Resource Statistics</h4>
                        <table class="table table-striped table-hover">
                            <tbody>
                                <tr>
                                    <td class="fw-bold">Most Used QOS</td>
                                    <td>{{ stats.top_qos|default('N/A') }}</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Most Used Partition</td>
                                    <td>{{ stats.top_partition|default('N/A') }}</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Unique Nodes Used</td>
                                    <td>{{ stats.unique_nodes_used|default('N/A')|int }}</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Avg GPUs per Job</td>
                                    <td>{{ "%.2f"|format(stats.avg_gpus_per_job|default(0)) }}</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold">Avg Memory per Job</td>
                                    <td>{{ "%.1f GB"|format(stats.avg_memory_request_gb|default(0)) }}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <footer>
                <div class="text-center mb-4">
                    <button class="pdf-button" id="exportPdfBottom">
                        <i class="fas fa-file-pdf"></i> Export to PDF
                    </button>
                </div>
                <p>Report generated on {{ current_date }}</p>
                <p>Enhanced Slurm Usage Report Generator</p>
            </footer>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        
        <script>
            // Chart.js configurations
            Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
            Chart.defaults.font.size = 14;
            Chart.defaults.color = '#2c3e50';
            
            // Common chart options
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            boxWidth: 10
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        titleFont: {
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 13
                        },
                        padding: 15,
                        cornerRadius: 5,
                        displayColors: true
                    }
                }
            };
            
            // Define chart colors
            const chartColors = [
                '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', 
                '#1abc9c', '#d35400', '#34495e', '#16a085', '#c0392b',
                '#2980b9', '#8e44ad', '#27ae60', '#e67e22', '#f1c40f'
            ];
            
            // Function to initialize all charts
            function initCharts() {
                // Job States Chart (Pie)
                if (document.getElementById('jobStatesChart') && {{ 'true' if 'job_states' in chart_data else 'false' }}) {
                    const jobStatesData = {{ chart_data.get('job_states', {}) | tojson }};
                    new Chart(document.getElementById('jobStatesChart'), {
                        type: 'pie',
                        data: {
                            labels: jobStatesData.labels,
                            datasets: [{
                                data: jobStatesData.data,
                                backgroundColor: chartColors.slice(0, jobStatesData.labels.length),
                                borderWidth: 1
                            }]
                        },
                        options: {
                            ...commonOptions,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Job State Distribution',
                                    font: {
                                        size: 18
                                    }
                                }
                            },
                            cutout: '30%'
                        }
                    });
                }
                
                // Jobs Over Time Chart (Line)
                if (document.getElementById('jobsOverTimeChart') && {{ 'true' if 'jobs_over_time' in chart_data else 'false' }}) {
                    const timelineData = {{ chart_data.get('jobs_over_time', {}) | tojson }};
                    new Chart(document.getElementById('jobsOverTimeChart'), {
                        type: 'line',
                        data: {
                            labels: timelineData.labels,
                            datasets: [{
                                label: 'Job Submissions',
                                data: timelineData.data,
                                borderColor: '#3498db',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.2,
                                pointRadius: 3,
                                pointHoverRadius: 5
                            }]
                        },
                        options: {
                            ...commonOptions,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    ticks: {
                                        maxRotation: 45,
                                        minRotation: 45
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Number of Jobs'
                                    }
                                }
                            },
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Job Submissions Over Time',
                                    font: {
                                        size: 18
                                    }
                                },
                                zoom: {
                                    pan: {
                                        enabled: true,
                                        mode: 'x'
                                    },
                                    zoom: {
                                        wheel: {
                                            enabled: true
                                        },
                                        pinch: {
                                            enabled: true
                                        },
                                        mode: 'x'
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Jobs Per User Chart (Horizontal Bar)
                if (document.getElementById('jobsPerUserChart') && {{ 'true' if 'jobs_per_user' in chart_data else 'false'}}) {
                    const userJobsData = {{ chart_data.get('jobs_per_user', {}) | tojson }};
                    new Chart(document.getElementById('jobsPerUserChart'), {
                        type: 'bar',
                        data: {
                            labels: userJobsData.labels,
                            datasets: [{
                                label: 'Number of Jobs',
                                data: userJobsData.data,
                                backgroundColor: chartColors.slice(0, userJobsData.labels.length),
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            indexAxis: 'y',
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Number of Jobs'
                                    }
                                },
                                y: {
                                    grid: {
                                        display: false
                                    },
                                    ticks: {
                                        autoSkip: false,
                                    }
                                }
                            },
                            maintainAspectRatio: false,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Jobs per User (Top 15)',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Completion Rate By User Chart (Horizontal Bar)
                if (document.getElementById('completionRateChart') && {{ 'true' if 'completion_rate_by_user' in chart_data else 'false' }}) {
                    const completionData = {{ chart_data.get('completion_rate_by_user', {}) | tojson }};
                    new Chart(document.getElementById('completionRateChart'), {
                        type: 'bar',
                        data: {
                            labels: completionData.labels,
                            datasets: [{
                                label: 'Completion Rate (%)',
                                data: completionData.data,
                                backgroundColor: completionData.data.map(value => {
                                    // Use a single color (green) with varying opacity based on completion rate
                                    // This creates a more consistent visual progression
                                    const baseColor = '52, 152, 219';  // blue
                                    const opacity = Math.max(0.3, value / 100); // Minimum opacity of 0.3
                                    return `rgba(${baseColor}, ${opacity})`;
                                }),
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            indexAxis: 'y',
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    max: 100,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Completion Rate (%)'
                                    }
                                },
                                y: {
                                    grid: {
                                        display: false
                                    },
                                    ticks: {
                                        autoSkip: false,
                                    }
                                }
                            },
                            maintainAspectRatio: false,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Job Completion Rate by User',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Jobs By Hour Chart (Bar)
                if (document.getElementById('jobsByHourChart') && {{ 'true' if 'jobs_by_hour' in chart_data else 'false' }}) {
                    const hourlyData = {{ chart_data.get('jobs_by_hour', {}) | tojson }};
                    new Chart(document.getElementById('jobsByHourChart'), {
                        type: 'bar',
                        data: {
                            labels: hourlyData.labels.map(hour => `${hour}:00`),
                            datasets: [{
                                label: 'Job Submissions',
                                data: hourlyData.data,
                                backgroundColor: '#3498db',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'Hour of Day'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Number of Jobs'
                                    }
                                }
                            },
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Job Submissions by Hour of Day',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Jobs By Weekday Chart (Bar)
                if (document.getElementById('jobsByWeekdayChart') && {{ 'true' if 'jobs_by_weekday' in chart_data else 'false'}}) {
                    const weekdayData = {{ chart_data.get('jobs_by_weekday', {}) | tojson }};
                    new Chart(document.getElementById('jobsByWeekdayChart'), {
                        type: 'bar',
                        data: {
                            labels: weekdayData.labels,
                            datasets: [{
                                label: 'Job Submissions',
                                data: weekdayData.data,
                                backgroundColor: '#2ecc71',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'Day of Week'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Number of Jobs'
                                    }
                                }
                            },
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Job Submissions by Day of Week',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // QOS Usage Chart (Doughnut)
                if (document.getElementById('qosUsageChart') && {{ 'true' if 'qos_usage' in chart_data else 'false'}}) {
                    const qosData = {{ chart_data.get('qos_usage', {}) | tojson }};
                    new Chart(document.getElementById('qosUsageChart'), {
                        type: 'doughnut',
                        data: {
                            labels: qosData.labels,
                            datasets: [{
                                data: qosData.data,
                                backgroundColor: chartColors.slice(0, qosData.labels.length),
                                borderWidth: 1
                            }]
                        },
                        options: {
                            ...commonOptions,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'QOS Usage Distribution',
                                    font: {
                                        size: 18
                                    }
                                }
                            },
                            cutout: '50%'
                        }
                    });
                }
                
                // GPU Usage Chart (Horizontal Bar)
                if (document.getElementById('gpuUsageChart') && {{ 'true' if 'gpu_usage' in chart_data else 'false'}}) {
                    const gpuData = {{ chart_data.get('gpu_usage', {}) | tojson }};
                    new Chart(document.getElementById('gpuUsageChart'), {
                        type: 'bar',
                        data: {
                            labels: gpuData.labels,
                            datasets: [{
                                label: 'GPU Hours',
                                data: gpuData.data,
                                backgroundColor: '#e74c3c',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            indexAxis: 'y',
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'GPU Hours'
                                    }
                                },
                                y: {
                                    grid: {
                                        display: false,
                                    },
                                    ticks: {
                                        autoSkip: false,
                                    }
                                }
                            },
                            maintainAspectRatio: false,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'GPU Hours per User (Top 15)',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // CPU Usage Chart (Horizontal Bar)
                if (document.getElementById('cpuUsageChart') && {{ 'true' if 'cpu_usage' in chart_data else 'false'}}) {
                    const cpuData = {{ chart_data.get('cpu_usage', {}) | tojson }};
                    new Chart(document.getElementById('cpuUsageChart'), {
                        type: 'bar',
                        data: {
                            labels: cpuData.labels,
                            datasets: [{
                                label: 'CPU Hours',
                                data: cpuData.data,
                                backgroundColor: '#9b59b6',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            indexAxis: 'y',
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'CPU Hours'
                                    }
                                },
                                y: {
                                    grid: {
                                        display: false
                                    },
                                    ticks: {
                                        autoSkip: false,
                                    }
                                }
                            },
                            maintainAspectRatio: false,
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'CPU Hours per User (Top 15)',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Job Runtime Boxplot Chart (custom)
                if (document.getElementById('jobRuntimeChart') && {{ 'true' if 'job_runtime_boxplot' in chart_data else 'false'}}) {
                    const runtimeData = {{ chart_data.get('job_runtime_boxplot', {}) | tojson }};
                    
                    // Extract QOS labels
                    const qosLabels = Object.keys(runtimeData);
                    
                    // Prepare data with error bars
                    const medianData = qosLabels.map(qos => ({
                        y: runtimeData[qos].median,
                        yMin: Math.max(0, runtimeData[qos].median - runtimeData[qos].std),
                        yMax: runtimeData[qos].median + runtimeData[qos].std
                    }));
                    
                    const meanData = qosLabels.map(qos => ({
                        y: runtimeData[qos].mean,
                        yMin: Math.max(0, runtimeData[qos].mean - runtimeData[qos].std),
                        yMax: runtimeData[qos].mean + runtimeData[qos].std
                    }));

                    new Chart(document.getElementById('jobRuntimeChart'), {
                        type: 'barWithErrorBars',
                        data: {
                            labels: qosLabels,
                            datasets: [
                                {
                                    label: 'Average Runtime (hours)',
                                    data: meanData,
                                    backgroundColor: 'rgba(46, 204, 113, 0.7)',
                                    borderColor: 'rgba(46, 204, 113, 1.0)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'QOS'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Job Runtime (hours)'
                                    }
                                }
                            },
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Job Runtime by QOS (with Standard Deviation)',
                                    font: {
                                        size: 18
                                    }
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(tooltipItems) {
                                            return tooltipItems[0].label + ' QOS';
                                        },
                                        label: function(tooltipItem) {
                                            const index = tooltipItem.dataIndex;
                                            const qos = qosLabels[index];
                                            const value = tooltipItem.parsed.y;
                                            const std = runtimeData[qos].std;
                                            
                                            return [
                                                `${tooltipItem.dataset.label}: ${value.toFixed(2)} hrs`,
                                                `Standard Deviation: ±${std.toFixed(2)} hrs`,
                                                `Count: ${runtimeData[qos].count} jobs`
                                            ];
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Memory Usage Chart (Bar)
                if (document.getElementById('memoryUsageChart') && {{ 'true' if 'memory_per_qos' in chart_data else 'false'}}) {
                    const memoryData = {{ chart_data.get('memory_per_qos', {}) | tojson }};
                    new Chart(document.getElementById('memoryUsageChart'), {
                        type: 'bar',
                        data: {
                            labels: memoryData.labels,
                            datasets: [{
                                label: 'Average Memory (GB)',
                                data: memoryData.data,
                                backgroundColor: '#1abc9c',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'QOS'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Average Memory Request (GB)'
                                    }
                                }
                            },
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Average Memory Request per QOS',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Queue Time Chart (Bar)
                if (document.getElementById('queueTimeChart') && {{ 'true' if 'queue_time_qos' in chart_data else 'false'}}) {
                    const queueTimeData = {{ chart_data.get('queue_time_qos', {}) | tojson }};
                    new Chart(document.getElementById('queueTimeChart'), {
                        type: 'bar',
                        data: {
                            labels: queueTimeData.labels,
                            datasets: [{
                                label: 'Average Queue Time (minutes)',
                                data: queueTimeData.data,
                                backgroundColor: '#f39c12',
                                borderWidth: 0
                            }]
                        },
                        options: {
                            ...commonOptions,
                            scales: {
                                x: {
                                    grid: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'QOS'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    },
                                    title: {
                                        display: true,
                                        text: 'Queue Time (minutes)'
                                    }
                                }
                            },
                            plugins: {
                                ...commonOptions.plugins,
                                title: {
                                    display: true,
                                    text: 'Average Queue Time per QOS',
                                    font: {
                                        size: 18
                                    }
                                }
                            }
                        }
                    });
                }
            }
            
            // Initialize all charts when the document is ready
            document.addEventListener('DOMContentLoaded', initCharts);


            // PDF export functionality
            document.addEventListener('DOMContentLoaded', function() {             
                // Replace the existing exportToPdf function with this enhanced version:
                const exportToPdf = function() {
                    
                    // Create a print message
                    const printMsg = document.createElement('div');
                    printMsg.style.position = 'fixed';
                    printMsg.style.top = '20px';
                    printMsg.style.left = '50%';
                    printMsg.style.transform = 'translateX(-50%)';
                    printMsg.style.backgroundColor = 'rgba(0,0,0,0.8)';
                    printMsg.style.color = 'white';
                    printMsg.style.padding = '15px 20px';
                    printMsg.style.borderRadius = '5px';
                    printMsg.style.zIndex = '9999';
                    printMsg.style.textAlign = 'center';
                    printMsg.innerHTML = '<p>Save as PDF using your browser print dialog.<br>When the print dialog opens, select "Save as PDF" option.</p>';
                    document.body.appendChild(printMsg);

                    // Give time for message to display, then print
                    setTimeout(() => {
                        window.print();
                        // Remove the message after print dialog closes
                        setTimeout(() => {
                            document.body.removeChild(printMsg);
                        }, 1000);
                    }, 1000);
                };
                
                // Attach event listeners to buttons
                document.getElementById('exportPdfTop').addEventListener('click', exportToPdf);
                document.getElementById('exportPdfBottom').addEventListener('click', exportToPdf);
            });

        </script>
    </body>
</html>
    """)

    # Render the template with our data
    html_content = template.render(
        chart_data=chart_data,
        stats=stats,
        qos_settings=qos_settings,
        user_qos=user_qos,
        args=args,
        current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return html_content

def main():
    """Main function to run the report generation process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Get QOS settings
    print("Fetching QOS settings...")
    qos_settings = get_qos_settings()
    
    # Get User QOS assignments
    print("Fetching user QOS assignments...")
    user_qos = get_user_qos()
    
    # Run sacct command
    print(f"Fetching Slurm accounting data...")
    sacct_output = run_sacct_command(
        start_date=args.start_date,
        end_date=args.end_date,
        user=args.user,
        partition=args.partition
    )
    
    if not sacct_output:
        print("Error: No data returned from sacct command.")
        return
    
    # Process the data
    print("Processing job data...")
    df = process_data(sacct_output)
    
    if df.empty:
        print("Error: No valid job data to process.")
        return
    
    # Prepare chart data
    print("Preparing chart data...")
    chart_data = prepare_chart_data(df, args.exclude_qos)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(df)
    
    # Generate HTML report
    print("Creating HTML report...")
    html_content = generate_html_report(chart_data, stats, qos_settings, user_qos, args)
    
    # Save the report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report successfully generated: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
