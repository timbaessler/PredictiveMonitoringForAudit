# Necessary imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import norm, gamma
import math
import sys
sys.path.append('..')
from config.data_config import syn_dict as bpi_dict
from src.preprocessing.utils import *
import os

plt.rcParams.update({'text.usetex':True,
                     'font.size':11,
                     'font.family':'serif'
                     })


def calculate_log_statistics(log_path):
    # Read the log file
    log = pd.read_feather(log_path)

    # Calculate statistics
    n_cases = log['CaseID'].nunique()
    positive_class_ratio = log.groupby('CaseID')['Late'].last().mean()
    n_numerical_attributes = log.select_dtypes(include=[np.number]).shape[1]
    n_case_attributes = len([col for col in log.columns if log[col].nunique() == n_cases])
    n_cat_event_attributes = len([col for col in log.columns if log[col].dtype == 'object' and col not in ['CaseID', 'Activity']])
    n_activity_classes = log['Activity'].nunique()
    median_trace_length = log.groupby('CaseID').size().median()

    # Create a formatted string for the statistics
    stats_string = f"""
\\begin{{table}}[H]
\\begin{{tabular}}{{ccccccc}}

  \\# Cases &
  \\begin{{tabular}}[c]{{@{{}}l@{{}}}}Positive \\\\ class ratio\\end{{tabular}} &
  \\begin{{tabular}}[c]{{@{{}}l@{{}}}}\\# Numerical \\\\ attributes  \\end{{tabular}} &
  \\begin{{tabular}}[c]{{@{{}}l@{{}}}}\\# Case\\\\ attributes \\\\ \\end{{tabular}} &
  \\begin{{tabular}}[c]{{@{{}}l@{{}}}}\\# Cat. event \\\\ attributes\\\\ \\end{{tabular}} &
   \\begin{{tabular}}[c]{{@{{}}l@{{}}}}\\# Activity \\\\ classes \\\\ \\end{{tabular}} &
      \\begin{{tabular}}[c]{{@{{}}l@{{}}}}Median \\\\trace length \\\\ \\end{{tabular}}  \\\\ \\hline

  {n_cases:,} &
  {positive_class_ratio:.2%} &
  {n_numerical_attributes} &
  {n_case_attributes} &
  {n_cat_event_attributes}  &
  {n_activity_classes} &
  {median_trace_length:.0f}\\\\ \\hline
\\end{{tabular}}
\\caption{{Statistics of the event logs and selected attributes.}}
\\label{{logstats}}
\\end{{table}}
"""

    return stats_string



class VendorProfile:
    def __init__(self, vendor_id):
        self.vendor_id = f"Vendor_{vendor_id:03d}"
        self.base_reliability = random.uniform(0.7, 0.99)
        self.seasonal_pattern = random.choice(['start', 'mid', 'end', 'none'])
        self.preferred_term = random.choice([30, 45, 60])
        self.term_distribution = {
            30: 0.8 if self.preferred_term == 30 else 0.1,
            45: 0.8 if self.preferred_term == 45 else 0.1,
            60: 0.8 if self.preferred_term == 60 else 0.1
        }
        self.daily_volume_mean = random.uniform(1, 5)
        self.daily_volume_std = self.daily_volume_mean * 0.3
        self.typical_amount_mean = random.uniform(1000, 8000)
        self.typical_amount_std = self.typical_amount_mean * 0.2
        self.historical_late_ratio = random.betavariate(2, 8)
        self.sector = random.choice(['IT', 'Manufacturing', 'Services', 'Retail'])
        self.complexity_factor = {
            'IT': 1.2,
            'Manufacturing': 1.5,
            'Services': 1.0,
            'Retail': 0.8
        }[self.sector]

class ApproverProfile:
    def __init__(self, name, seniority):
        self.name = name
        self.seniority = seniority
        self.processing_speed = random.uniform(0.8, 1.2)
        self.base_accuracy = random.uniform(0.9, 0.99)
        self.experience_years = {
            'junior': random.randint(1, 3),
            'senior': random.randint(4, 8),
            'manager': random.randint(8, 15)
        }[seniority]
        self.peak_hour = random.randint(9, 16)
        self.efficiency_curve = norm(self.peak_hour, 2)
        self.max_daily_capacity = {
            'junior': 20,
            'senior': 30,
            'manager': 15
        }[seniority]
        self.approval_limits = {
            'junior': 5000,
            'senior': 20000,
            'manager': float('inf')
        }[seniority]
        self.specialized_sectors = random.sample(
            ['IT', 'Manufacturing', 'Services', 'Retail'],
            random.randint(1, 3)
        )
        self.sector_efficiency = {
            sector: random.uniform(1.1, 1.3) if sector in self.specialized_sectors
            else random.uniform(0.8, 1.0)
            for sector in ['IT', 'Manufacturing', 'Services', 'Retail']
        }

def calculate_payment_run(due_date):
    """Find the nearest Friday after or on the given date"""
    if pd.isna(due_date):
        return np.nan
    temp_date = due_date
    while temp_date.weekday() != 4:  # 4 represents Friday
        temp_date += timedelta(days=1)
    return temp_date


def select_approver(approvers_dict, vendor_profile, invoice_amount,
                    current_workloads, excluded_approver=None):
    """Select most appropriate approver based on multiple factors"""
    suitable_approvers = []

    for name, approver in approvers_dict.items():
        if name == excluded_approver:
            continue

        if invoice_amount <= approver.approval_limits:
            workload_score = 1 - (current_workloads[name] / approver.max_daily_capacity)
            sector_score = approver.sector_efficiency[vendor_profile.sector]
            experience_score = math.log(approver.experience_years + 1) / 5

            suitability = (workload_score * 0.4 +
                           sector_score * 0.3 +
                           experience_score * 0.3)

            suitable_approvers.append((name, suitability))

    if not suitable_approvers:
        return random.choice(list(approvers_dict.keys()))

    total_suitability = sum(score for _, score in suitable_approvers)
    probabilities = [score/total_suitability for _, score in suitable_approvers]

    return random.choices(
        [name for name, _ in suitable_approvers],
        weights=probabilities
    )[0]

def calculate_second_approval_delay(remaining_time, scenario, payment_term=None):
    """Calculate realistic second approval delay based on scenario with mixed distribution"""
    if scenario == 'urgent':
        return random.randint(1, 2)
    elif scenario == 'last_minute':
        return max(1, remaining_time - random.randint(1, 2))
    elif scenario == 'holiday':
        return min(remaining_time - 2, random.randint(3, 5))
    else:
        # We need at least 2 business days for payment processing after second approval
        safe_remaining_time = remaining_time - 2

        if safe_remaining_time <= 0:
            return 1  # Emergency fast-track if very little time left

        # Single mixed distribution for all payment terms
        choices = [
            # Early approvals (20%)
            1, 1, 1, 2, 2, 2, 3, 3, 3,
            # Mid-range approvals (30%)
            4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            # Later approvals (20%)
            8, 8, 9, 9, 10, 10, 11, 11,
            # Very late approvals (20%)
            12, 12, 13, 13, 14, 14, 15, 15,
            # Outliers (10%)
            16, 17, 18, 19, 20
        ]

        # Add some randomization to the weights themselves
        base_weights = [
            # Early approvals
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            # Mid-range
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            # Later
            2, 2, 2, 2, 2, 2, 2, 2,
            # Very late
            2, 2, 2, 2, 2, 2, 2, 2,
            # Outliers
            1, 1, 1, 1, 1
        ]

        # Add random variation to weights
        weights = [w * random.uniform(0.8, 1.2) for w in base_weights]

        # Additional randomization: occasionally boost early or late approvals
        if random.random() < 0.3:  # 30% chance of shifting distribution
            if random.random() < 0.5:
                # Boost early approvals
                weights[:9] = [w * 1.5 for w in weights[:9]]
            else:
                # Boost late approvals
                weights[-13:] = [w * 1.5 for w in weights[-13:]]

        delay = random.choices(choices, weights=weights)[0]

        # Add small random variation to the selected delay
        if delay > 1 and random.random() < 0.3:  # 30% chance of variation
            delay += random.choice([-1, 1])

        return max(1, min(delay, safe_remaining_time))

def generate_invoice_case(case_id, start_date, vendor_profiles, approver_profiles,
                          current_workloads, invoice_amount_range, payment_terms, is_late_case):
    """Generate a single invoice case with realistic patterns"""
    case_id_str = f"INV-{case_id:05d}"
    events = []

    # Increased variation in urgency probabilities
    is_urgent = random.random() < 0.05
    is_last_minute = random.random() < 0.12  # Slightly increased
    has_technical_issues = random.random() < 0.04  # Slightly increased
    is_holiday_period = random.random() < 0.08

    vendor_profile = random.choice(list(vendor_profiles.values()))
    invoice_amount = random.gauss(vendor_profile.typical_amount_mean,
                                  vendor_profile.typical_amount_std)
    invoice_amount = max(invoice_amount_range[0],
                         min(invoice_amount_range[1], invoice_amount))

    if is_urgent:
        payment_term = min(payment_terms)
    elif is_holiday_period:
        payment_term = max(payment_terms)
    else:
        payment_term = random.choice(payment_terms)

    # More variable initial delay based on payment term
    if payment_term <= 7:
        max_initial_delay = max(1, payment_term - 3)  # Allow more variation for short terms
    else:
        # More variation in initial delay for longer terms
        max_initial_delay = max(1, payment_term - random.randint(0, 15))

    invoice_date = start_date + timedelta(days=random.randint(0, max_initial_delay))
    due_date = invoice_date + timedelta(days=payment_term)

    events.append([case_id_str, vendor_profile.vendor_id, invoice_date,
                   "Invoice Created", invoice_amount, payment_term, None, False, due_date])

    if has_technical_issues:
        booking_delay = random.randint(3, 6)  # More variation
        events.append([case_id_str, vendor_profile.vendor_id,
                       invoice_date + timedelta(days=2),
                       "System Error", invoice_amount, payment_term, None, False, due_date])
    elif is_urgent:
        booking_delay = 1
    else:
        booking_delay = random.randint(1, 4)  # More variation

    current_date = invoice_date + timedelta(days=booking_delay)
    events.append([case_id_str, vendor_profile.vendor_id, current_date,
                   "Invoice Booked", invoice_amount, payment_term, "Paul", False, due_date])

    if not is_late_case:
        total_time_available = (due_date - current_date).days

        # More variable first approval timing
        if is_urgent:
            target_first_approval = random.randint(1, 2)
        elif is_last_minute:
            target_first_approval = max(1, total_time_available - random.randint(3, 6))
        elif has_technical_issues:
            target_first_approval = random.randint(2, 5)  # More variation
        else:
            # More variation in normal cases
            min_remaining_time = random.randint(10, 25)  # Wider range
            target_first_approval = max(1, total_time_available - min_remaining_time)

        first_approval_date = current_date + timedelta(days=max(1, target_first_approval))
        first_approver_name = select_approver(approver_profiles, vendor_profile,
                                              invoice_amount, current_workloads)
        events.append([case_id_str, vendor_profile.vendor_id, first_approval_date,
                       "First Approval Complete", invoice_amount, payment_term,
                       first_approver_name, False, due_date])

        remaining_time = (due_date - first_approval_date).days
        scenario = 'urgent' if is_urgent else \
            'last_minute' if is_last_minute else \
                'holiday' if is_holiday_period else 'normal'

        # Pass payment_term to calculate_second_approval_delay
        second_approval_delay = calculate_second_approval_delay(remaining_time, scenario, payment_term)
        second_approval_date = first_approval_date + timedelta(days=second_approval_delay)

        # Safety check: ensure second approval is before due date with enough time for payment
        if (due_date - second_approval_date).days < 2:
            second_approval_date = due_date - timedelta(days=2)

        second_approver_name = select_approver(approver_profiles, vendor_profile,
                                               invoice_amount, current_workloads,
                                               excluded_approver=first_approver_name)

        events.append([case_id_str, vendor_profile.vendor_id, second_approval_date,
                       "Second Approval Complete", invoice_amount, payment_term,
                       second_approver_name, False, due_date])

        # Payment process
        payment_schedule_date = second_approval_date + timedelta(days=1)
        payment_date = calculate_payment_run(payment_schedule_date)

        if payment_date <= due_date:
            events.append([case_id_str, vendor_profile.vendor_id, payment_schedule_date,
                           "Payment Scheduled", invoice_amount, payment_term,
                           None, False, due_date])
            events.append([case_id_str, vendor_profile.vendor_id, payment_date,
                           "Payment Executed", invoice_amount, payment_term,
                           None, False, due_date])
        else:
            is_late_case = True

    if is_late_case:
        if has_technical_issues:
            first_approval_date = due_date + timedelta(days=random.randint(2, 5))  # More variation
            second_approval_date = first_approval_date + timedelta(days=random.randint(1, 3))
        elif is_holiday_period:
            first_approval_date = due_date + timedelta(days=random.randint(1, 4))
            second_approval_date = first_approval_date + timedelta(days=random.randint(2, 5))
        else:
            if random.random() < 0.7:
                first_approval_date = due_date - timedelta(days=random.randint(1, 3))
                second_approval_date = due_date + timedelta(days=random.randint(1, 3))
            else:
                first_approval_date = due_date + timedelta(days=random.randint(1, 4))
                second_approval_date = first_approval_date + timedelta(days=random.randint(1, 3))

        first_approver_name = select_approver(approver_profiles, vendor_profile,
                                              invoice_amount, current_workloads)
        events.append([case_id_str, vendor_profile.vendor_id, first_approval_date,
                       "First Approval Complete", invoice_amount, payment_term,
                       first_approver_name, True, due_date])

        second_approver_name = select_approver(approver_profiles, vendor_profile,
                                               invoice_amount, current_workloads,
                                               excluded_approver=first_approver_name)
        events.append([case_id_str, vendor_profile.vendor_id, second_approval_date,
                       "Second Approval Complete", invoice_amount, payment_term,
                       second_approver_name, True, due_date])

        payment_schedule_date = second_approval_date + timedelta(days=1)
        payment_date = calculate_payment_run(payment_schedule_date)

        events.append([case_id_str, vendor_profile.vendor_id, payment_schedule_date,
                       "Payment Scheduled", invoice_amount, payment_term,
                       None, True, due_date])
        events.append([case_id_str, vendor_profile.vendor_id, payment_date,
                       "Payment Executed", invoice_amount, payment_term,
                       None, True, due_date])

    return events

def analyze_approval_distribution(log):
    """Analyze and visualize the distribution of second approvals before deadline"""
    second_approvals = log[
        (log['Activity'] == 'Second Approval Complete') &
        (~log['Late'])
        ].copy()

    second_approvals['business_days_until_deadline'] = second_approvals.apply(
        lambda row: len(pd.date_range(row['Timestamp'], row['deadline'],
                                      freq='B')) - 1, axis=1
    )

    distribution = second_approvals['business_days_until_deadline'].value_counts().sort_index()
    print("\nDistribution of business days until deadline for second approvals:")
    print(distribution)

    print("\nApproval timing statistics:")
    print(second_approvals['business_days_until_deadline'].describe())

    ranges = [(0,3), (4,7), (8,11), (12,15)]
    print("\nPercentage of approvals by time ranges:")
    for start, end in ranges:
        mask = (second_approvals['business_days_until_deadline'] >= start) & \
               (second_approvals['business_days_until_deadline'] <= end)
        percentage = (mask.sum() / len(second_approvals)) * 100
        print(f"{start}-{end} days: {percentage:.1f}%")

        plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 6))

    sns.histplot(
        data=second_approvals,
        x='business_days_until_deadline',
        bins=20,
        #ax=ax1,
        color='#2878B5'
    )
    plt.title('Distribution of Second Approvals\nBefore Deadline', pad=20)
    plt.xlabel('Business Days Until Deadline')
    plt.ylabel('Frequency')


    #plt.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=10)
    #plt.xlabel(plt.xlabel(), fontsize=20)
    #plt.ylabel(plt.ylabel(), fontsize=20)
    #ax.set_title(ax.get_title(), fontsize=20)
    plt.rcParams.update({'text.usetex':True,
                         'font.size':11,
                         'font.family':'serif'
                         })
    plt.tight_layout()
    plt.savefig(os.path.join(bpi_dict["res_path"], "last_begin_payment_syn.png"), dpi=500)
    plt.close()


if __name__ == "__main__":
    # Configuration
    n_cases = 50000
    start_date = datetime(2024, 1, 1)
    invoice_amount_range = (100, 10000)
    payment_terms = [7, 15, 30]
    target_positive_class_ratio = 0.17

    # Initialize profiles
    vendor_profiles = {f"Vendor_{i:03d}": VendorProfile(i) for i in range(1, 501)}

    approver_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
    approver_profiles = {}
    for name in approver_names[:3]:
        approver_profiles[name] = ApproverProfile(name, 'manager')
    for name in approver_names[3:6]:
        approver_profiles[name] = ApproverProfile(name, 'senior')
    for name in approver_names[6:]:
        approver_profiles[name] = ApproverProfile(name, 'junior')

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Generate cases
    late_cases = int(n_cases * target_positive_class_ratio)
    late_case_ids = random.sample(range(1, n_cases + 1), late_cases)
    all_events = []
    current_workloads = {name: 0 for name in approver_names}

    for case_id in range(1, n_cases + 1):
        is_late_case = case_id in late_case_ids
        case_events = generate_invoice_case(
            case_id, start_date, vendor_profiles, approver_profiles,
            current_workloads, invoice_amount_range, payment_terms, is_late_case
        )
        all_events.extend(case_events)

        if case_id % 100 == 0:
            current_workloads = {name: 0 for name in approver_names}

    # Create DataFrame
    columns = ["CaseID", "Vendor", "Timestamp", "Activity", "Invoice Amount",
               "Payment Term", "Resource", "Late", "deadline"]
    log = pd.DataFrame(all_events, columns=columns)
    log.sort_values(by=["CaseID", "Timestamp"], inplace=True)

    # Add process mining columns
    timestamp_col="Timestamp"
    case_id_col = "CaseID"
    # Add derived features
    log = get_time_since_last_event(log, timestamp_col=timestamp_col, case_id_col=case_id_col)
    log = get_time_since_first_event(log, timestamp_col=timestamp_col, case_id_col=case_id_col)
    #log = get_event_nr(log, case_id_col=case_id_col)
    #log = get_event_duration(log, timestamp_col=timestamp_col, case_id_col=case_id_col)
    #log = get_remaining_time(log, timestamp_col=timestamp_col, case_id_col=case_id_col)
    #log = get_total_duration(log, timestamp_col=timestamp_col, case_id_col=case_id_col)
    #log = get_seq_length(log, case_id_col=case_id_col).reset_index(drop=True)
    log['payment_run'] = log['deadline'].apply(calculate_payment_run)
    log["Late"] = log.groupby(case_id_col)["Late"].transform("last")
    cols = [case_id_col] + [timestamp_col] + bpi_dict["static_cat_cols"] + bpi_dict["dynamic_cat_cols"] + bpi_dict["num_cols"] + ["Late", "deadline"]


    print(cols)
    log = log[cols]
    log.to_feather(bpi_dict["labelled_log_path"])
    log.to_csv(bpi_dict["csv_path"], index=False, sep=";")

    # Print statistics and create visualizations
    print("\nDataset Summary:")
    print(f"Total cases: {log['CaseID'].nunique():,}")
    print(f"Total events: {len(log):,}")
    print(f"Average events per case: {len(log) / log['CaseID'].nunique():.2f}")
    act_classes = log["Activity"].nunique()
    print(f"Number of Activity Classes:{act_classes}")
    med_log = np.median(log.groupby("CaseID").size())
    print(f"Median event length: {med_log}")
    log_agg = log.groupby(["CaseID"], as_index=False)["Late"].last()
    positive_class_ratio = log_agg["Late"].mean()
    print("PCR", positive_class_ratio)


