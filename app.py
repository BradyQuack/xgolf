#Version_021 - In progress - Role Configuration - New schedule algorithmn
#app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import time
import base64
import io
from matplotlib.backends.backend_pdf import PdfPages
import traceback

st.set_page_config(page_title="AI Shift Optimizer", layout="wide")

# Initialize session state for shift configurations
if 'shift_config' not in st.session_state:
    st.session_state.shift_config = {
        'Shift 1': {'name': 'Morning Shift', 'start': 9, 'end': 16},
        'Shift 2': {'name': 'Evening Shift', 'start': 16, 'end': 24}
    }

# Initialize roles if not present
if 'roles_config' not in st.session_state:
    st.session_state.roles_config = {
        'Role 1': {'name': 'Bartender', 'color': 'blue', 'optimize': True},
        'Role 2': {'name': 'General Help', 'color': 'green', 'optimize': False}
    }

# Initialize staff configuration if not present
if 'staff_config' not in st.session_state:
    st.session_state.staff_config = {
        'Role 1': 1,  # Default staff count for Role 1
        'Role 2': 1   # Default staff count for Role 2
    }

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process CSV data with caching for efficiency."""
    try:
        data = pd.read_csv(uploaded_file)
        df = data[['Date', 'Time', 'Gross Sales', 'Employee']].copy()
        
        # Clean and process data
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        df['Weekday'] = df['Date'].dt.day_name()
        df['Hour'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S').dt.hour
        df['Gross Sales'] = df['Gross Sales'].replace(r'[\$,]', '', regex=True).astype(float)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Filter out any invalid weekday entries
        valid_weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        df = df[df['weekday'].str.lower().isin(valid_weekdays)]
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error(traceback.format_exc())
        return None

##################################################################################################################################################################################################################################################

def validate_shift_times(start, end, shift_key):
    """Validate shift start and end times."""
    if end <= start:
        st.sidebar.error(f"⚠️ {shift_key}: End time must be after start time.")
        return False
    
    # Check for overlapping shifts with other shifts
    for key, shift in st.session_state.shift_config.items():
        if key == shift_key:
            continue
        
        # Check for overlap
        if (start < shift['end'] and end > shift['start']):
            st.sidebar.warning(f"⚠️ {shift_key} overlaps with {key}. This may cause scheduling conflicts.")
    
    return True

##################################################################################################################################################################################################################################################

def configure_roles():
    """Create a dedicated sidebar section for role configuration with updated layout."""
    st.sidebar.header("Role Configuration")
    
    with st.sidebar.expander("⚙️ Role Settings", expanded=False):
        # Display all existing roles
        role_keys = sorted(
            st.session_state.roles_config.keys(),
            key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0
        )
        
        for role_key in role_keys:
            role_data = st.session_state.roles_config[role_key]
            
            st.write(f"**{role_key}**")
            
            # Create columns with focus on Role Name and Optimize checkbox
            cols = st.columns([3, 1])
            
            with cols[0]:
                # Role name input
                name = st.text_input(
                    "Role Name",
                    value=role_data['name'],
                    key=f'role_name_{role_key}'  # Modified key to ensure uniqueness
                )
            
            # Performance optimization checkbox - SINGLE CHECKBOX
            optimize = st.checkbox(
                "Optimize for Performance",
                value=role_data.get('optimize', True),
                key=f'role_optimize_{role_key}',  # Modified key to ensure uniqueness
                help="When checked, this role will be filled with the highest performing employees based on efficiency scores. If unchecked, scheduling will prioritize equal distribution of shifts."
            )
            
            # Remove button - don't allow removing the default roles
            if role_key not in ['Role 1', 'Role 2']:
                if st.button(f"❌ Remove", key=f'remove_role_{role_key}'):
                    try:
                        del st.session_state.roles_config[role_key]
                        
                        # Also remove this role from employee availability preferences
                        if 'availability' in st.session_state:
                            for emp in st.session_state.availability:
                                if 'roles' in st.session_state.availability[emp] and role_key in st.session_state.availability[emp]['roles']:
                                    del st.session_state.availability[emp]['roles'][role_key]
                        
                        # Remove from staff config
                        if role_key in st.session_state.staff_config:
                            del st.session_state.staff_config[role_key]
                        
                        st.rerun()
                        return
                    except Exception as e:
                        st.error(f"Error removing role: {str(e)}")
            
            # Update role data
            st.session_state.roles_config[role_key] = {
                'name': name,
                'color': role_data.get('color', 'gray'),
                'optimize': optimize
            }
            
            # Add a separator between roles
            st.markdown("---")


##################################################################################################################################################################################################################################################

def configure_staff():
    """Create a dedicated sidebar section for staff count configuration."""
    st.sidebar.header("Staff Configuration")
    
    with st.sidebar.expander("⚙️ Staff Settings", expanded=False):
        # Display staff settings for each role
        role_keys = sorted(
            st.session_state.roles_config.keys(),
            key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0
        )
        
        for role_key in role_keys:
            role_data = st.session_state.roles_config[role_key]
            role_name = role_data['name']
            
            cols = st.columns([3, 1])
            
            with cols[0]:
                st.write(f"**{role_name}**")
            
            with cols[1]:
                # Staff count input
                staff_count = st.number_input(
                    "Staff",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.staff_config.get(role_key, 1),
                    key=f'staff_{role_key}',
                    step=1
                )
                # Update staff count in session state
                st.session_state.staff_config[role_key] = staff_count
            
            # Add a separator between roles
            st.markdown("---")

##################################################################################################################################################################################################################################################

def configure_shifts():
    """Create interactive shift configuration controls with proper ordering, validation and fixed add button."""
    st.sidebar.header("Shift Configuration")
    
    with st.sidebar.expander("⚙️ Shift Settings", expanded=False):
        # Create a container for all shifts (except the add button)
        shifts_container = st.container()
        
        # Create a separate container at the bottom for the add button
        button_container = st.container()
        
        with shifts_container:
            # Get sorted shift keys (numerically by shift number)
            shift_keys = sorted(
                st.session_state.shift_config.keys(),
                key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0
            )
            
            for shift_key in shift_keys:
                shift_data = st.session_state.shift_config[shift_key]
                
                st.write(f"**{shift_key}**")
                
                # Name row
                name = st.text_input(
                    "Shift Name",
                    value=shift_data['name'],
                    key=f'{shift_key}_name'
                )
                
                # Time row
                cols1 = st.columns([1, 1])
                with cols1[0]:
                    start = st.number_input(
                        "Start",
                        min_value=0,
                        max_value=23,
                        value=shift_data['start'],
                        key=f'{shift_key}_start',
                        step=1
                    )
                with cols1[1]:
                    end = st.number_input(
                        "End",
                        min_value=1,
                        max_value=24,
                        value=shift_data['end'],
                        key=f'{shift_key}_end',
                        step=1
                    )
                
                # Role and Staff row
                st.write("**Role Staffing**")
                
                # Display each role with a staff count input
                for role_key, role_data in st.session_state.roles_config.items():
                    role_cols = st.columns([3, 1])
                    with role_cols[0]:
                        st.write(f"{role_data['name']}")
                    
                    with role_cols[1]:
                        # Get existing staff count for this role in this shift
                        role_staff = shift_data.get('role_staff', {}).get(role_key, 1)
                        
                        # Staff count input
                        staff_count = st.number_input(
                            "Staff",
                            min_value=0,
                            max_value=10,
                            value=role_staff,
                            key=f'{shift_key}_{role_key}_staff',
                            step=1
                        )
                        
                        # Store the staff count for this role
                        if 'role_staff' not in shift_data:
                            shift_data['role_staff'] = {}
                        
                        shift_data['role_staff'][role_key] = staff_count
                
                # Validate times
                is_valid = validate_shift_times(start, end, shift_key)
                
                # Remove button
                remove_col = st.columns([5, 1])
                with remove_col[1]:
                    # Don't show remove button for the first two shifts
                    if shift_key not in ['Shift 1', 'Shift 2']:
                        if st.button(f"❌ Remove", key=f'remove_{shift_key}'):
                            try:
                                del st.session_state.shift_config[shift_key]
                                st.rerun()
                                return
                            except Exception as e:
                                st.error(f"Error removing shift: {str(e)}")
                
                # Only update if valid
                if is_valid:
                    # Update the shift configuration while preserving role_staff data
                    updated_shift = {
                        'name': name,
                        'start': start,
                        'end': end,
                        'staff': shift_data.get('staff', 2),  # Keep default staff
                        'role_staff': shift_data.get('role_staff', {})  # Preserve role staff counts
                    }
                    st.session_state.shift_config[shift_key] = updated_shift
                
                # Add a separator between shifts
                st.markdown("---")
        
        # Always show the add button at the bottom
        with button_container:
            st.write("**Shift Management**")
            if st.button("➕ Add Another Shift"):
                try:
                    # Find the next available shift number
                    existing_numbers = [
                        int(key.split()[1]) for key in st.session_state.shift_config.keys()
                        if key.split()[1].isdigit()
                    ]
                    next_num = max(existing_numbers) + 1 if existing_numbers else 3
                    
                    new_shift_name = f'Shift {next_num}'
                    
                    # Initialize with role_staff for all existing roles
                    role_staff = {}
                    for role_key in st.session_state.roles_config:
                        role_staff[role_key] = 1  # Default staff count per role
                    
                    st.session_state.shift_config[f'Shift {next_num}'] = {
                        'name': new_shift_name,
                        'start': 0,
                        'end': 8,
                        'staff': 2,  # Default staff count
                        'role_staff': role_staff  # Add role-specific staff counts
                    }
                    
                    # Initialize availability for the new shift for all employees
                    if 'availability' in st.session_state:
                        for emp in st.session_state.availability:
                            st.session_state.availability[emp]['shifts'][new_shift_name] = True
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding shift: {str(e)}")

##################################################################################################################################################################################################################################################

def get_employee_availability(df):
    """Create interactive availability input form for employees."""
    st.sidebar.header("Employee Availability")
    
    try:
        # Get unique employees sorted by efficiency
        employee_sales = df.groupby('employee')['gross_sales'].sum().sort_values(ascending=False)
        employees = employee_sales.index.tolist()
        
        # Initialize availability dictionary in session state
        if 'availability' not in st.session_state:
            st.session_state.availability = {}
        
        for emp in employees:
            # Initialize default availability if not set
            if emp not in st.session_state.availability:
                st.session_state.availability[emp] = {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    'shifts': {shift['name']: True for shift in st.session_state.shift_config.values()},
                    'roles': {role: True for role in st.session_state.roles_config},  # Add roles preference
                    'max_shifts': 3  # Default value
                }
            
            # Ensure all current shifts are in employee's availability
            for shift in st.session_state.shift_config.values():
                if shift['name'] not in st.session_state.availability[emp]['shifts']:
                    st.session_state.availability[emp]['shifts'][shift['name']] = True
            
            # Ensure all current roles are in employee's preferences
            if 'roles' not in st.session_state.availability[emp]:
                st.session_state.availability[emp]['roles'] = {}
            for role in st.session_state.roles_config:
                if role not in st.session_state.availability[emp]['roles']:
                    st.session_state.availability[emp]['roles'][role] = True
            
            with st.sidebar.expander(f"⏰ {emp} (${employee_sales[emp]:,.0f})"):
                # Max shifts input
                max_shifts = st.number_input(
                    "Max Shifts Per Week",
                    min_value=1,
                    max_value=7,
                    value=st.session_state.availability[emp].get('max_shifts', 5),
                    key=f"{emp}_max_shifts"
                )

                # Shift checkboxes - dynamically generated based on configured shifts
                st.write("**Shift Availability**")
                shift_prefs = {}
                for shift_key, shift in st.session_state.shift_config.items():
                    shift_prefs[shift['name']] = st.checkbox(
                        f"{shift['name']} ({shift['start']}-{shift['end']})",
                        value=st.session_state.availability[emp]['shifts'].get(shift['name'], True),
                        key=f"{emp}_{shift_key}"
                    )
                
                # Role preference checkboxes with actual role names displayed
                st.write("**Role Preferences**")
                role_prefs = {}
                for role_key, role_data in st.session_state.roles_config.items():
                    # Use the role's display name rather than the role_key
                    display_name = role_data['name']
                    role_prefs[role_key] = st.checkbox(
                        display_name,
                        value=st.session_state.availability[emp]['roles'].get(role_key, True),
                        key=f"{emp}_{role_key}"
                    )

                # Days multiselect
                st.write("**Available Days**")
                days = st.multiselect(
                    "Days",
                    options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    default=st.session_state.availability[emp]['days'],
                    key=f"{emp}_days"
                )
                
                # Validate at least one day is selected
                if not days:
                    st.warning(f"⚠️ {emp} must be available for at least one day.")
                
                # Update session state when inputs change
                if (days != st.session_state.availability[emp]['days'] or 
                    any(shift_prefs[shift['name']] != st.session_state.availability[emp]['shifts'].get(shift['name'], True)
                    for shift in st.session_state.shift_config.values()) or
                    any(role_prefs[role_key] != st.session_state.availability[emp]['roles'].get(role_key, True)
                    for role_key in st.session_state.roles_config) or
                    max_shifts != st.session_state.availability[emp].get('max_shifts', 5)):
                    
                    st.session_state.availability[emp] = {
                        'days': days,
                        'shifts': shift_prefs,
                        'roles': role_prefs,
                        'max_shifts': max_shifts
                    }
                    st.rerun()
        
        return st.session_state.availability
    except Exception as e:
        st.error(f"Error processing employee availability: {str(e)}")
        st.error(traceback.format_exc())
        return {}

##################################################################################################################################################################################################################################################

def assign_shifts(row):
    """Assign shift label based on hour and configured shifts."""
    try:
        hour = row['hour']
        for shift in st.session_state.shift_config.values():
            if shift['start'] <= hour < shift['end']:
                return shift['name']
        return 'Other'
    except Exception as e:
        st.error(f"Error assigning shifts: {str(e)}")
        return 'Unknown'

##################################################################################################################################################################################################################################################

def assign_shifts(row):
    """Assign shift label based on hour and configured shifts."""
    try:
        hour = row['hour']
        for shift in st.session_state.shift_config.values():
            if shift['start'] <= hour < shift['end']:
                return shift['name']
        return 'Other'
    except Exception as e:
        st.error(f"Error assigning shifts: {str(e)}")
        return 'Unknown'

##################################################################################################################################################################################################################################################

def plot_weekly_schedule_with_availability(df, availability):
    """
    Generate optimized schedule considering:
    - Shift configurations (start/end times)
    - Role requirements (staff per role from shift's role_staff)
    - Employee availability (days, shifts, roles, max shifts)
    - Employee efficiency in specific shifts (for optimized roles)
    - Equal distribution of shifts (for non-optimized roles)
    - Shift sales (higher revenue shifts prioritized)
    """
    try:
        # Set up weekday order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
        
        # Get shift names
        morning_shift = st.session_state.shift_config['Shift 1']['name']
        evening_shift = st.session_state.shift_config['Shift 2']['name']
        
        # Assign shift type based on hour
        df['shift_type'] = df.apply(assign_shifts, axis=1)
        
        # Create separate DataFrames for morning and evening shift data
        morning_data = df[df['shift_type'] == morning_shift]
        evening_data = df[df['shift_type'] == evening_shift]
        
        # Calculate efficiency scores for shifts
        morning_scores = calculate_shift_efficiency_scores(morning_data)
        evening_scores = calculate_shift_efficiency_scores(evening_data)
        
        # Process shift data
        current_shifts = list(st.session_state.shift_config.values())
        
        # Calculate shift importance based on historical sales
        shift_importance = {}
        for shift in current_shifts:
            shift_hours = range(shift['start'], shift['end'])
            shift_sales = df[df['hour'].isin(shift_hours)].groupby('weekday')['gross_sales'].sum()
            shift_importance[shift['name']] = shift_sales.reindex(weekday_order, fill_value=0)
        
        # Ensure all shifts have an importance value
        for shift in current_shifts:
            if shift['name'] not in shift_importance:
                shift_importance[shift['name']] = pd.Series(0, index=weekday_order)
        
        # Create scheduling structures
        schedule = {}
        for day in weekday_order:
            schedule[day] = {}
            for shift in current_shifts:
                schedule[day][shift['name']] = {}
                for role_key in st.session_state.roles_config:
                    schedule[day][shift['name']][role_key] = []
        
        # Display schedule DataFrame
        display_schedule = pd.DataFrame(index=weekday_order)
        for shift in current_shifts:
            display_schedule[shift['name']] = ''
        
        # Create prioritized shift queue (highest sales first)
        shift_queue = []
        for day in weekday_order:
            for shift_name, shift_data in shift_importance.items():
                shift_queue.append((day, shift_name, shift_data[day]))
        
        # Sort by sales descending
        shift_queue.sort(key=lambda x: -x[2])
        
        # Initialize employee assignments
        employee_assignments = {}
        if morning_scores is not None:
            for emp in morning_scores.index:
                if emp not in employee_assignments:
                    employee_assignments[emp] = []
        
        if evening_scores is not None:
            for emp in evening_scores.index:
                if emp not in employee_assignments:
                    employee_assignments[emp] = []
        
        # Assign employees to shifts and roles
        for day, shift_name, _ in shift_queue:
            shift_config = next((s for s in current_shifts if s['name'] == shift_name), None)
            if not shift_config:
                continue
            
            # Get role requirements for this shift
            for role_key, role_data in st.session_state.roles_config.items():
                # Get staff count from shift's role_staff configuration
                required_staff = 0
                if 'role_staff' in shift_config and role_key in shift_config['role_staff']:
                    required_staff = shift_config['role_staff'].get(role_key, 0)
                
                if required_staff <= 0:
                    continue  # Skip roles with no staff requirement
                
                current_staff = schedule[day][shift_name][role_key]
                
                # Skip if role already fully staffed
                if len(current_staff) >= required_staff:
                    continue
                
                # Select the appropriate efficiency scores based on shift type
                if shift_name == morning_shift and morning_scores is not None:
                    efficiency_scores = morning_scores
                elif shift_name == evening_shift and evening_scores is not None:
                    efficiency_scores = evening_scores
                else:
                    # Fallback to overall sales
                    efficiency_scores = df.groupby('employee')['gross_sales'].sum().sort_values(ascending=False)
                
                # Find qualified employees with availability for THIS shift AND role
                available_employees = []
                for emp in efficiency_scores.index:
                    if emp in availability and (
                        day in availability[emp]['days'] and
                        availability[emp]['shifts'].get(shift_name, False) and
                        availability[emp]['roles'].get(role_key, False) and  # Check role preference
                        len(employee_assignments[emp]) < availability[emp].get('max_shifts', 2) and
                        emp not in current_staff and
                        not any(a['day'] == day for a in employee_assignments[emp])
                    ):
                        available_employees.append(emp)
                
                # Check if this role should be optimized
                role_optimize = st.session_state.roles_config[role_key].get('optimize', True)
                
                if role_optimize:
                    # For optimized roles, sort available employees by efficiency score (descending)
                    available_employees.sort(key=lambda x: efficiency_scores.loc[x] if x in efficiency_scores.index else 0, 
                                            reverse=True)
                else:
                    # For non-optimized roles, sort available employees by number of assigned shifts (ascending)
                    # This gives preference to employees with fewer assigned shifts
                    available_employees.sort(key=lambda x: len(employee_assignments[x]))
                
                # Assign needed staff
                needed = required_staff - len(current_staff)
                for emp in available_employees[:needed]:
                    assignment = {
                        'day': day,
                        'shift': shift_name,
                        'role': role_key,
                        'hours': shift_config['end'] - shift_config['start']
                    }
                    employee_assignments[emp].append(assignment)
                    schedule[day][shift_name][role_key].append(emp)
                    
                    # Update display schedule
                    current_cell = display_schedule.at[day, shift_name]
                    role_name = st.session_state.roles_config[role_key]['name']
                    # Get employee's score for this shift
                    score = efficiency_scores.loc[emp] if emp in efficiency_scores.index else 0
                    emp_display = f"{role_name} - {emp} - Score: {score:.0f}"
                    display_schedule.at[day, shift_name] = f"{current_cell}\n\n{emp_display}".strip() if current_cell else emp_display
        
        # Store schedule in session state for export
        st.session_state.schedule_df = display_schedule.copy()
        
        # Create visualization
        shift_columns = [shift['name'] for shift in current_shifts]
        dummy_data = np.ones((len(display_schedule), len(shift_columns)))
        
        # Format labels for heatmap
        shift_labels = []
        for col in shift_columns:
            col_labels = []
            for day in weekday_order:
                cell_content = display_schedule.at[day, col]
                if '\n\n' in cell_content:
                    parts = cell_content.split('\n\n')
                    formatted = '\n'.join([p.replace('\n', ' ') for p in parts])
                    col_labels.append(formatted)
                else:
                    col_labels.append(cell_content.replace('\n', ' '))
            shift_labels.append(col_labels)
        
        employee_labels = np.array(shift_labels).T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            dummy_data,
            annot=employee_labels,
            fmt='',
            cmap='Greens',
            linewidths=1,
            linecolor='gray',
            cbar=False,
            annot_kws={
                'fontsize': 14 if any('\n\n' in cell for cell in display_schedule.values.flatten()) else 16,
                'ha': 'center',
                'va': 'center',
                'fontweight': 'bold'
            },
            ax=ax
        )
        
        ax.set_xticklabels(shift_columns, rotation=0, fontsize=16, fontweight='bold')
        ax.set_yticklabels(display_schedule.index, rotation=0, fontsize=16, fontweight='bold')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title('AI Optimized Labor Schedule (Role & Shift Efficiency)', pad=24, fontsize=22, fontweight='bold')
        plt.tight_layout()
        
        # Save figure for PDF export
        st.session_state.schedule_fig = fig
        
        return fig
    except Exception as e:
        st.error(f"Error generating schedule: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating schedule: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

##################################################################################################################################################################################################################################################

# Helper function to calculate efficiency scores for a shift
# Helper function to calculate efficiency scores for a shift
def calculate_shift_efficiency_scores(shift_data):
    """Calculate efficiency scores for employees in a specific shift."""
    if shift_data.empty:
        return None
    
    # === METRIC 1: TOTAL SALES SCORE ===
    # Calculate total sales by employee
    total_sales = shift_data.groupby('employee')['gross_sales'].sum().sort_values(ascending=False)
    
    if len(total_sales) == 0:
        return None
    
    # Create scoring based on position (highest sales = max score, lowest = 1)
    num_employees = len(total_sales)
    total_sales_score = pd.Series(
        range(num_employees, 0, -1), 
        index=total_sales.index
    )
    
    # === METRIC 2: AVERAGE SALE PER TRANSACTION SCORE ===
    # Fixed calculation to avoid KeyError with avg_sale
    transaction_counts = shift_data.groupby('employee')['date'].count()
    avg_sale_values = total_sales / transaction_counts
    avg_sale_values = avg_sale_values.sort_values(ascending=False)
    
    # Create scoring based on position
    avg_sale_score = pd.Series(
        range(num_employees, 0, -1),
        index=avg_sale_values.index
    )
    
    # === METRIC 3: TOTAL SHIFTS WORKED SCORE ===
    # Count unique shifts worked per employee
    unique_shifts = shift_data.drop_duplicates(['employee', 'date'])
    shifts_worked = unique_shifts.groupby('employee').size().sort_values(ascending=False)
    
    # Create scoring based on position
    shifts_worked_score = pd.Series(
        range(num_employees, 0, -1),
        index=shifts_worked.index
    )
    
    # === NEW METRIC 4: AVERAGE SALES PER SHIFT SCORE ===
    # Calculate average sales per shift for each employee
    avg_sales_per_shift = total_sales / shifts_worked
    avg_sales_per_shift = avg_sales_per_shift.sort_values(ascending=False)
    
    # Create scoring based on position
    avg_sales_per_shift_score = pd.Series(
        range(num_employees, 0, -1),
        index=avg_sales_per_shift.index
    )
    
    # === COMBINE SCORES ===
    # Create DataFrame with all scores
    scores = pd.DataFrame({
        'Total Sales Score': total_sales_score,
        'Avg Sale Score': avg_sale_score,
        'Shifts Worked Score': shifts_worked_score,
        'Avg Sales Per Shift Score': avg_sales_per_shift_score  # Add new metric
    })
    
    # Calculate sum of scores
    total_score = scores.sum(axis=1)
    
    return total_score

##################################################################################################################################################################################################################################################

@st.cache_data
def generate_shift_analysis(df):
    """Generate shift efficiency analysis based on current shift config."""
    try:
        heatmap_data = df.pivot_table(
            values='gross_sales',
            index='weekday',
            columns='hour',
            aggfunc='sum',
            fill_value=0
        )
        
        shift_results = {}
        for shift_key, shift in st.session_state.shift_config.items():
            shift_hours = list(range(shift['start'], shift['end']))
            
            # Only include hours that exist in the data
            available_hours = [h for h in shift_hours if h in heatmap_data.columns]
            
            if available_hours:
                shift_results[shift['name']] = heatmap_data[available_hours].sum(axis=1)
            else:
                # If no hours match, create empty series with correct index
                shift_results[shift['name']] = pd.Series(0, index=heatmap_data.index)
        
        return pd.DataFrame(shift_results)
    except Exception as e:
        st.error(f"Error analyzing shifts: {str(e)}")
        return pd.DataFrame()
        
##################################################################################################################################################################################################################################################

@st.cache_data
def analyze_employee_shift_efficiency(df):
    """Analyze employee efficiency by shift."""
    try:
        # Assign shift to each row based on hour
        df['shift'] = df.apply(assign_shifts, axis=1)
        
        # Calculate metrics by employee and shift
        employee_shift_stats = df.groupby(['employee', 'shift']).agg({
            'gross_sales': 'sum',
            'date': 'count'  # Count transactions
        }).rename(columns={'date': 'transaction_count'})
        
        # Calculate average sale per transaction
        employee_shift_stats['avg_sale'] = employee_shift_stats['gross_sales'] / employee_shift_stats['transaction_count']
        
        return employee_shift_stats
    except Exception as e:
        st.error(f"Error analyzing employee shift efficiency: {str(e)}")
        return pd.DataFrame()

##################################################################################################################################################################################################################################################

def get_csv_download_link():
    """Generate a CSV download link for the schedule."""
    try:
        if 'schedule_df' in st.session_state and not st.session_state.schedule_df.empty:
            csv = st.session_state.schedule_df.to_csv()
            b64 = base64.b64encode(csv.encode()).decode()
            return f'data:file/csv;base64,{b64}'
        return None
    except Exception as e:
        st.error(f"Error generating CSV link: {str(e)}")
        return None

##################################################################################################################################################################################################################################################

def get_excel_download_link():
    """Generate an Excel download link for the schedule."""
    try:
        if 'schedule_df' in st.session_state and not st.session_state.schedule_df.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.schedule_df.to_excel(writer, sheet_name='Schedule')
                
                # Format the Excel file
                workbook = writer.book
                worksheet = writer.sheets['Schedule']
                
                # Add a header format
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Write the column headers with the header format
                for col_num, value in enumerate(st.session_state.schedule_df.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)
            
            b64 = base64.b64encode(output.getvalue()).decode()
            return f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
        return None
    except Exception as e:
        st.error(f"Error generating Excel link: {str(e)}")
        return None

##################################################################################################################################################################################################################################################

def get_pdf_download_link():
    """Generate a PDF download link for the schedule."""
    try:
        if 'schedule_fig' in st.session_state:
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                pdf.savefig(st.session_state.schedule_fig)
            
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return f'data:application/pdf;base64,{b64}'
        return None
    except Exception as e:
        st.error(f"Error generating PDF link: {str(e)}")
        return None

##################################################################################################################################################################################################################################################

def plot_employee_shift_type_count(df):
    """
    Generate a heatmap showing the top employees who worked each shift type the most.
    Counts each employee only once per day/shift combination, regardless of transaction count.
    Displays with shift types on y-axis and days on x-axis.
    """
    try:
        # Ensure proper weekday ordering
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
        
        # Assign shift type based on hour
        df['shift_type'] = df.apply(assign_shifts, axis=1)
        
        # Get the shift names from the configuration
        morning_shift = st.session_state.shift_config['Shift 1']['name']
        evening_shift = st.session_state.shift_config['Shift 2']['name']
        shift_types = [morning_shift, evening_shift]
        
        # Add a slider to control how many top employees to display for each shift
        num_employees = st.slider(
            "Number of top employees to display per shift",
            min_value=1, 
            max_value=10,
            value=3,  # Default to 3
            key="shift_type_count_slider"
        )
        
        # Create a DataFrame with the proper structure for our heatmap - ROTATED AXES
        # Now shift types are on y-axis and days are on x-axis
        heatmap_data = pd.DataFrame(index=shift_types, columns=weekday_order)
        
        # Create a numeric data matrix for the heatmap colors - ROTATED AXES
        numeric_data = np.zeros((len(shift_types), len(weekday_order)))
        
        # For each day and shift combination, find the unique employees and their shift counts
        for i, shift in enumerate(shift_types):
            for j, day in enumerate(weekday_order):
                # Get unique employees for this specific day and shift
                day_shift_data = df[(df['weekday'] == day) & (df['shift_type'] == shift)]
                
                # Count shifts by unique employee
                # First, get unique (employee, date) combinations to count actual shifts
                unique_shifts = day_shift_data.drop_duplicates(['employee', 'date'])
                employee_shift_counts = unique_shifts.groupby('employee').size().sort_values(ascending=False)
                
                # Get the top N employees for this day/shift
                top_day_shift_employees = employee_shift_counts.head(num_employees)
                
                # Format the cell content with employee names and counts
                cell_content = ""
                for emp, count in top_day_shift_employees.items():
                    if count > 0:  # Only show employees who worked shifts
                        cell_content += f"{emp} : {count}\n"
                
                # Store in the heatmap DataFrame - ROTATED AXES
                heatmap_data.at[shift, day] = cell_content.strip() if cell_content else "0"
                
                # Store the number of unique employees who worked this day/shift for the heat coloring
                numeric_data[i, j] = len(day_shift_data['employee'].unique())
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(16, 6))  # Adjusted figsize for rotated orientation
        
        # Create the heatmap with numeric data for colors
        hm = sns.heatmap(
            numeric_data,
            cmap='Greens',
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Unique Employees'},
            ax=ax
        )
        
        # Overlay text annotations manually - ROTATED AXES
        for i, shift in enumerate(shift_types):
            for j, day in enumerate(weekday_order):
                text = heatmap_data.at[shift, day]
                ax.text(
                    j + 0.5,  # Center of the cell horizontally
                    i + 0.5,  # Center of the cell vertically
                    text,
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='normal'
                )
        
        # Configure plot appearance - ROTATED AXES
        ax.set_title(f'Top {num_employees} Employees by Shift Count', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Set tick labels - ROTATED AXES
        ax.set_xticklabels(weekday_order, rotation=0, fontsize=12, fontweight='bold')
        ax.set_yticklabels(shift_types, rotation=0, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating shift type count heatmap: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating heatmap: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

##################################################################################################################################################################################################################################################

def generate_shift_analysis_rotated(df):
    """Generate shift efficiency analysis based on current shift config with rotated layout."""
    try:
        # Ensure proper weekday ordering
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
        
        heatmap_data = df.pivot_table(
            values='gross_sales',
            index='weekday',
            columns='hour',
            aggfunc='sum',
            fill_value=0
        )
        
        # Get the shift names from the configuration
        morning_shift = st.session_state.shift_config['Shift 1']['name']
        evening_shift = st.session_state.shift_config['Shift 2']['name']
        shift_types = [morning_shift, evening_shift]
        
        # Calculate sales for each shift
        shift_results = {}
        for shift_key, shift in st.session_state.shift_config.items():
            shift_hours = list(range(shift['start'], shift['end']))
            
            # Only include hours that exist in the data
            available_hours = [h for h in shift_hours if h in heatmap_data.columns]
            
            if available_hours:
                shift_results[shift['name']] = heatmap_data[available_hours].sum(axis=1)
            else:
                # If no hours match, create empty series with correct index
                shift_results[shift['name']] = pd.Series(0, index=heatmap_data.index)
        
        # Create DataFrame from results
        shift_summary = pd.DataFrame(shift_results)
        
        # ROTATE THE LAYOUT - Transpose the DataFrame to have shifts on Y-axis and days on X-axis
        shift_summary_rotated = shift_summary.T
                
        # Create the visualization with rotated layout
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # First create the heatmap without annotations
        hm = sns.heatmap(
            shift_summary_rotated,
            cmap=['white'],
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Gross Sales ($)'},
            ax=ax,
            annot=False  # Don't let seaborn handle annotations
        )
        
        # Manually add the dollar annotations
        for i in range(shift_summary_rotated.shape[0]):
            for j in range(shift_summary_rotated.shape[1]):
                value = shift_summary_rotated.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f"${value:,.0f}", 
                        ha="center", va="center", fontsize=12, fontweight='bold')
        
        # Configure plot appearance
        ax.set_title('Sales by Shift and Weekday', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Set tick labels
        ax.set_xticklabels(weekday_order, rotation=0, fontsize=12, fontweight='bold')
        ax.set_yticklabels(shift_types, rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error analyzing shifts: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating shift analysis: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

##################################################################################################################################################################################################################################################

def plot_simplified_employee_morning_score(df):
    """
    Create a simplified scoring visualization for employees working the morning shift
    based on multiple efficiency metrics, showing only the total score.
    Now includes average sales per shift in the score calculation.
    """
    try:
        # Assign shift type based on hour
        df['shift_type'] = df.apply(assign_shifts, axis=1)
        
        # Get the morning shift name from configuration
        morning_shift = st.session_state.shift_config['Shift 1']['name']
        
        # Filter data for morning shift only
        morning_data = df[df['shift_type'] == morning_shift]
        
        # If no morning shift data, return early
        if morning_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data available for morning shift", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Calculate the total score using our updated function
        total_score = calculate_shift_efficiency_scores(morning_data)
        if total_score is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Insufficient data to calculate scores", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort the scores for visualization
        total_score = total_score.sort_values(ascending=False)
        
        # Add slider to control number of employees displayed
        num_top_employees = st.slider(
            "Number of employees to display",
            min_value=1,
            max_value=len(total_score),
            value=min(10, len(total_score)),
            key="morning_efficiency_slider"
        )
        
        # Filter to top N employees based on slider
        top_scores = total_score.head(num_top_employees)
        
        # Create horizontal bar chart for total scores only
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot horizontal bars for total score only
        bars = ax.barh(
            top_scores.index[::-1],  # Reverse order to show highest score at top
            top_scores.values[::-1],
            color='forestgreen',
            edgecolor='black',
            alpha=0.8
        )
        
        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.5,                      # Slightly to the right of the bar
                bar.get_y() + bar.get_height()/2, # Center of the bar
                f'{width:.0f}',                   # Format as integer
                va='center',                      # Vertical alignment
                fontweight='bold',                # Make text bold
                fontsize=12                       # Increase font size
            )
        
        # Configure chart appearance
        ax.set_title(f'Top {num_top_employees} Employee Efficiency - {morning_shift}', 
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('Efficiency Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        
        # Set axis limits to provide some padding
        max_score = total_score.max()
        ax.set_xlim(0, max_score * 1.1)  # Add 10% padding to the right
        
        # Add a grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # After creating the bar chart, add:
        ax.tick_params(axis='y', labelsize=16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        
        # Improve overall appearance
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating employee efficiency score: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig


##################################################################################################################################################################################################################################################

def plot_simplified_employee_evening_score(df):
    """
    Create a simplified scoring visualization for employees working the evening shift
    based on multiple efficiency metrics, showing only the total score.
    Now includes average sales per shift in the score calculation.
    """
    try:
        # Assign shift type based on hour
        df['shift_type'] = df.apply(assign_shifts, axis=1)
        
        # Get the evening shift name from configuration
        evening_shift = st.session_state.shift_config['Shift 2']['name']
        
        # Filter data for evening shift only
        evening_data = df[df['shift_type'] == evening_shift]
        
        # If no evening shift data, return early
        if evening_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data available for evening shift", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Calculate the total score using our updated function
        total_score = calculate_shift_efficiency_scores(evening_data)
        if total_score is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Insufficient data to calculate scores", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort the scores for visualization
        total_score = total_score.sort_values(ascending=False)
        
        # Add slider to control number of employees displayed
        num_top_employees = st.slider(
            "Number of employees to display",
            min_value=1,
            max_value=len(total_score),
            value=min(10, len(total_score)),
            key="evening_efficiency_slider"
        )
        
        # Filter to top N employees based on slider
        top_scores = total_score.head(num_top_employees)
        
        # Create horizontal bar chart for total scores only
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot horizontal bars for total score only
        bars = ax.barh(
            top_scores.index[::-1],  # Reverse order to show highest score at top
            top_scores.values[::-1],
            color='darkgreen',  # Darker green to distinguish from morning shift
            edgecolor='black',
            alpha=0.8
        )
        
        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.5,                      # Slightly to the right of the bar
                bar.get_y() + bar.get_height()/2, # Center of the bar
                f'{width:.0f}',                   # Format as integer
                va='center',                      # Vertical alignment
                fontweight='bold',                # Make text bold
                fontsize=12                       # Increase font size
            )
        
        # Configure chart appearance
        ax.set_title(f'Top {num_top_employees} Employee Efficiency - {evening_shift}', 
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('Efficiency Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        
        # Set axis limits to provide some padding
        max_score = total_score.max()
        ax.set_xlim(0, max_score * 1.1)  # Add 10% padding to the right
        
        # Add a grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # After creating the bar chart, add:
        ax.tick_params(axis='y', labelsize=16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        
        # Improve overall appearance
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating employee efficiency score: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig


##################################################################################################################################################################################################################################################

def plot_avg_sales_per_shift(df):
    """
    Create a visualization showing average sales per shift by employee.
    This divides total sales by the number of shifts each employee worked.
    """
    try:
        
        # First, get unique (employee, date) combinations to count actual shifts
        # This counts each employee only once per day regardless of transaction count
        unique_shifts = df.drop_duplicates(['employee', 'date'])
        
        # Count shifts per employee
        shifts_worked = unique_shifts.groupby('employee').size()
        
        # Calculate total sales per employee
        total_sales = df.groupby('employee')['gross_sales'].sum()
        
        # Calculate average sales per shift
        avg_sales_per_shift = total_sales / shifts_worked
        avg_sales_per_shift = avg_sales_per_shift.sort_values(ascending=False)
        
        # Add slider to control number of employees displayed
        num_employees = st.slider(
            "Number of employees to display",
            min_value=1,
            max_value=len(avg_sales_per_shift),
            value=min(10, len(avg_sales_per_shift)),
            key="avg_sales_per_shift_slider"
        )
        
        # Filter to top N employees based on slider
        top_employees = avg_sales_per_shift.head(num_employees)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(16, 8))
        bars = top_employees.sort_values().plot(
            kind='barh',
            color='mediumseagreen',
            edgecolor='black',
            ax=ax
        )
        
        # Format x-axis with dollar signs
        ax.xaxis.set_major_formatter('${x:,.0f}')
        
        # Calculate maximum value for setting axis limits
        max_value = top_employees.max()
        # Set x-axis limit with 15% padding for labels
        ax.set_xlim(0, max_value * 1.15)
        
        # Add value labels to the end of each bar
        for i, v in enumerate(top_employees.sort_values()):
            ax.text(v + (max_value * 0.02), i, f"${v:,.0f}", va='center', fontweight='bold')
        
        # Configure chart appearance
        ax.set_xlabel('Average Sales per Shift ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Average Sales per Shift by Employee', fontsize=16, fontweight='bold')
        ax.tick_params(axis='y', labelsize=16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        
        # Add a grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Improve overall appearance
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating average sales per shift visualization: {str(e)}")
        st.error(traceback.format_exc())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

##################################################################################################################################################################################################################################################

# Main app
try:
    st.title("📈 AI Shift Optimization Dashboard")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Add Instructions expander
    with st.expander("📘 Instructions", expanded=True):
        st.markdown("""
        ### Overview
        The X-Golf Shift Optimizer is a data-driven tool designed to create optimized employee schedules based on:
        - Historical sales efficiency
        - Employee availability
        - Shift patterns
        
        **Key Benefit:** Places top-performing employees in the top-grossing shifts.

        ### How to Use
        1. **Data Upload**
           - Download Square monthly sales data
           - Click the "Browse files" button to upload sales data CSV
           - Data processing begins automatically after upload

        2. **Configure Shifts** (⚙️ Sidebar)
           - Default shifts:
             - Morning Shift (9 AM - 4 PM)
             - Evening Shift (4 PM - 12 AM)
           - Customizable:
             - Shift names
             - Start/end times
             - Staff requirements
           - Add shifts with ➕ button

        3. **Set Employee Availability** (⚙️ Sidebar)
           - Per employee settings:
             - Max shifts/week (1-7)
             - Available shifts
             - Available days

        4. **Explore Visualizations**
           - Sales patterns by hour & day
           - Shift efficiency comparisons
           - Employee efficiency rankings
           - AI-optimized schedule

        ### Efficiency Metrics
        | Metric | Description | Importance |
        |---|---|----|
        | Total Sales | Overall revenue generated | Experience indicator |
        | Avg Sale/Transaction | Average transaction value | Upselling ability indicator |
        | Shift Count | Number of shifts worked | Availability indicator |
        | Avg Sales/Shift | Sales performance per shift | Shift productivity indicator |
        | Efficiency Score | Combined metrics score | Employee efficiency indicator |

        ### New Feature: Average Sales Per Shift
        The new "Average Sales Per Shift" metric provides:
        - A balanced view of employee productivity per shift
        - Accounts for different shift frequencies
        - Helps identify employees who excel in specific shifts
        - Prevents total sales bias toward employees who simply work more shifts

        ### Optimization Benefits
        | Feature | Benefit | Impact |
        |---|---|----|
        | Shift-specific scheduling | Employees in best-performing shifts | 15-25% revenue potential |
        | Data-driven assignments | Removes scheduling bias | Fairer process |
        | Historical analysis | Optimal staffing levels | Cost reduction |
        | Efficiency tracking | Identify top performers | Better development |
        """)

    if uploaded_file:
        # Load and process data with caching
        df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Configure shifts and roles
        configure_roles()
        configure_shifts()     

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 1: WEEKDAY VS HOUR HEATMAP (COLLAPSIBLE) ===
        with st.expander("📊 Sales Heatmap by Weekday/Hour", expanded=False):
            # Ensure proper weekday ordering
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

            heatmap_data = df.pivot_table(
                values='gross_sales',
                index='weekday',
                columns='hour',
                aggfunc='sum',
                fill_value=0
            ).reindex(weekday_order)  # Force correct order

            fig1, ax1 = plt.subplots(figsize=(16, 8))
            sns.heatmap(
                heatmap_data,
                cmap='Greens',
                linewidths=0.3,
                linecolor='gray',
                fmt='',
                annot=np.array([["${:,.0f}".format(val) for val in row] for row in heatmap_data.values]),
                annot_kws={"size": 8},
                cbar_kws={'label': 'Gross Sales ($)'},
                ax=ax1
            )
            ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
            ax1.set_ylabel('')
            ax1.set_title('Sales by Weekday and Hour', fontsize=16, fontweight='bold')
            st.pyplot(fig1)

            st.markdown("""
            **🔍 Analysis Breakdown:**  
            - **Purpose:** Identify hourly sales patterns and peak revenue periods  
            - **Key Metrics:** Gross sales ($) aggregated by hour and weekday  
            - **Business Impact:** Optimize staffing during high-value periods  
            """)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 2: SHIFT efficiency ANALYSIS (COLLAPSIBLE) ===
        with st.expander("📊 Sales Heatmap by Shift", expanded=False):
            # st.markdown("### Original Layout")
            shift_summary = generate_shift_analysis(df)
            rotated_shift_fig = generate_shift_analysis_rotated(df)
            st.pyplot(rotated_shift_fig)

            st.markdown("""
    **🔍 Shift Efficiency Breakdown:**  
    - **Purpose:** Compare shift performance across days  
    - **Key Metrics:** Total sales per configured shift  
    - **Business Impact:** Validate shift timing effectiveness  
    - **Comparison:** Direct morning vs evening shift revenue analysis  
    """)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 3: TOTAL SALES BY EMPLOYEE (COLLAPSIBLE) ===
        with st.expander("💰 Total Sales by Employee", expanded=False):
            employee_sales = df.groupby('employee')['gross_sales'].sum().sort_values(ascending=False)
            
            # Add slider to control number of employees displayed
            num_employees = st.slider(
                "Number of employees to display",
                min_value=1,
                max_value=len(employee_sales),
                value=min(10, len(employee_sales)),
                key="total_sales_slider"
            )
            
            # Filter to top N employees based on slider
            top_employees = employee_sales.head(num_employees)
            
            fig3, ax3 = plt.subplots(figsize=(16, 8))
            top_employees.sort_values().plot(
                kind='barh',
                color='green',
                edgecolor='black',
                ax=ax3
            )
            ax3.set_xlabel('Gross Sales ($)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('')
            ax3.set_title(f'Total Sales by Top {num_employees} Employees', fontsize=16, fontweight='bold')
            ax3.tick_params(axis='y', labelsize=16)
            for tick in ax3.yaxis.get_major_ticks():
                tick.label1.set_fontweight('bold')
            
            # Format x-axis with dollar signs
            ax3.xaxis.set_major_formatter('${x:,.0f}')
            
            # Calculate maximum value for setting axis limits
            max_value = top_employees.max()
            # Set x-axis limit with 15% padding for labels
            ax3.set_xlim(0, max_value * 1.15)
            
            # Add value labels
            for i, v in enumerate(top_employees.sort_values()):
                ax3.text(v + (max_value * 0.02), i, f"${v:,.0f}", va='center', fontweight='bold')
            
            st.pyplot(fig3)

             st.markdown("""
    **🔍 Top Performer Analysis:**  
    - **Purpose:** Identify revenue generation leaders  
    - **Key Metric:** Gross sales ($) per employee  
    - **Business Impact:** Recognize high-value staff  
    - **Optimization:** Prioritize top performers for peak shifts  
    """)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 4: AVERAGE SALE PER Shift (COLLAPSIBLE) ===
        with st.expander("💰 Average Sales per Shift by Employee", expanded=False):
            avg_sales_per_shift_fig = plot_avg_sales_per_shift(df)
            st.pyplot(avg_sales_per_shift_fig) 

            st.markdown("""
    **🔍 Shift Productivity Analysis:**  
    - **Purpose:** Measure true shift efficiency  
    - **New Metric:** (Total sales) / (Shifts worked)  
    - **Benefit:** Eliminates shift frequency bias  
    - **Strategic Value:** Identifies consistently productive staff  
    
    *Pro Tip: Use this to identify underutilized high performers*  
    """)
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 5: AVERAGE SALE PER TRANSACTION (COLLAPSIBLE) ===
        with st.expander("💰 Average Sale per Transaction by Employee", expanded=False):
            # Calculate average sale per transaction for each employee
            avg_sales = df.groupby('employee').agg({
                'gross_sales': 'sum',
                'date': 'count'  # Count transactions
            })
            avg_sales['avg_sale'] = avg_sales['gross_sales'] / avg_sales['date']
            avg_sales = avg_sales.sort_values('avg_sale', ascending=False)
            
            # Add slider to control number of employees displayed
            num_avg_employees = st.slider(
                "Number of employees to display",
                min_value=1,
                max_value=len(avg_sales),
                value=min(10, len(avg_sales)),
                key="avg_sales_slider"
            )
            
            # Filter to top N employees based on slider
            top_avg_employees = avg_sales['avg_sale'].head(num_avg_employees)
            
            fig4, ax4 = plt.subplots(figsize=(16, 8))
            top_avg_employees.sort_values().plot(
                kind='barh',
                color='lightgreen',
                edgecolor='black',
                ax=ax4
            )
            ax4.set_xlabel('Average Sale ($)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('')
            ax4.set_title(f'Average Sale per Transaction - Top {num_avg_employees} Employees', fontsize=16, fontweight='bold')
            
            # Format x-axis with dollar signs
            ax4.xaxis.set_major_formatter('${x:,.0f}')
            
            # Calculate maximum value for setting axis limits
            max_value = top_avg_employees.max()
            # Set x-axis limit with 15% padding for labels
            ax4.set_xlim(0, max_value * 1.15)
            
            # Add value labels
            for i, v in enumerate(top_avg_employees.sort_values()):
                ax4.text(v + (max_value * 0.02), i, f"${v:,.0f}", va='center', fontweight='bold')

            ax4.tick_params(axis='y', labelsize=16)
            for tick in ax4.yaxis.get_major_ticks():
                tick.label1.set_fontweight('bold')
            
            st.pyplot(fig4)

            st.markdown("""
    **🔍 Upselling Effectiveness:**  
    - **Purpose:** Evaluate premium service capabilities  
    - **Key Metric:** Average transaction value ($)  
    - **Training Insight:** Identify coaching opportunities  
    - **VIP Impact:** Staff selection for high-value customers  
    
    *Pro Tip: Pair top performers with new hires for mentoring*  
    """)
        
        # Analyze shift-specific efficiency
        employee_shift_stats = analyze_employee_shift_efficiency(df)
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 6: TOTAL SALES BY EMPLOYEE - MORNING SHIFT (COLLAPSIBLE) ===
        morning_shift = st.session_state.shift_config['Shift 1']['name']
        with st.expander(f"👥 Total Sales by Employee - {morning_shift}", expanded=False):
            # Filter for Morning Shift
            try:
                morning_sales = employee_shift_stats.loc[(slice(None), morning_shift), 'gross_sales']
                morning_sales = morning_sales.reset_index().set_index('employee')['gross_sales'].sort_values(ascending=False)
                
                if not morning_sales.empty:
                    # Add slider to control number of employees displayed
                    num_morning_employees = st.slider(
                        "Number of employees to display",
                        min_value=1,
                        max_value=len(morning_sales),
                        value=min(10, len(morning_sales)),
                        key="morning_shift_slider"
                    )
                    
                    # Filter to top N employees based on slider
                    top_morning_employees = morning_sales.sort_values(ascending=False).head(num_morning_employees)
                    
                    fig5, ax5 = plt.subplots(figsize=(16, 8))
                    top_morning_employees.sort_values().plot(
                        kind='barh',
                        color='forestgreen',
                        edgecolor='black',
                        ax=ax5
                    )
                    ax5.set_xlabel('Gross Sales ($)', fontsize=12, fontweight='bold')
                    ax5.set_ylabel('')
                    ax5.set_title(f'Total Sales by Top {num_morning_employees} Employees - {morning_shift}', fontsize=16, fontweight='bold')
                    
                    # Format x-axis with dollar signs
                    ax5.xaxis.set_major_formatter('${x:,.0f}')
                    
                    # Calculate maximum value for setting axis limits
                    max_value = top_morning_employees.max()
                    # Set x-axis limit with 15% padding for labels
                    ax5.set_xlim(0, max_value * 1.15)
                    
                    # Add value labels
                    for i, v in enumerate(top_morning_employees.sort_values()):
                        ax5.text(v + (max_value * 0.02), i, f"${v:,.0f}", va='center', fontweight='bold')

                    ax5.tick_params(axis='y', labelsize=16)
                    for tick in ax5.yaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    
                    st.pyplot(fig5)

                    st.markdown(f"""
        **🔍 {shift_name} Performance:**  
        - **Purpose:** Identify shift-specific superstars  
        - **Key Metric:** Gross sales during {shift_name}  
        - **Scheduling Impact:** Match performers to optimal shifts  
        - **Pattern Recognition:** Reveal time-of-day strengths  
        
        *Pro Tip: Use this to create specialized shift teams*  
        """)

                else:
                    st.info(f"No data available for {morning_shift}")
            except Exception as e:
                st.error(f"Error generating {morning_shift} visualization: {str(e)}")
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 7: TOTAL SALES BY EMPLOYEE - EVENING SHIFT (COLLAPSIBLE) ===
        evening_shift = st.session_state.shift_config['Shift 2']['name']
        with st.expander(f"👥 Total Sales by Employee - {evening_shift}", expanded=False):
            # Filter for Evening Shift
            try:
                evening_sales = employee_shift_stats.loc[(slice(None), evening_shift), 'gross_sales']
                evening_sales = evening_sales.reset_index().set_index('employee')['gross_sales'].sort_values(ascending=False)
                
                if not evening_sales.empty:
                    # Add slider to control number of employees displayed
                    num_evening_employees = st.slider(
                        "Number of employees to display",
                        min_value=1,
                        max_value=len(evening_sales),
                        value=min(10, len(evening_sales)),
                        key="evening_shift_slider"
                    )
                    
                    # Filter to top N employees based on slider
                    top_evening_employees = evening_sales.head(num_evening_employees)
                    
                    fig6, ax6 = plt.subplots(figsize=(16, 8))
                    top_evening_employees.sort_values().plot(
                        kind='barh',
                        color='darkgreen',
                        edgecolor='black',
                        ax=ax6
                    )
                    ax6.set_xlabel('Gross Sales ($)', fontsize=12, fontweight='bold')
                    ax6.set_ylabel('')
                    ax6.set_title(f'Total Sales by Top {num_evening_employees} Employees - {evening_shift}', fontsize=16, fontweight='bold')

                    ax6.tick_params(axis='y', labelsize=16)
                    for tick in ax6.yaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    
                    # Format x-axis with dollar signs
                    ax6.xaxis.set_major_formatter('${x:,.0f}')
                    
                    # Calculate maximum value for setting axis limits
                    max_value = top_evening_employees.max()
                    # Set x-axis limit with 15% padding for labels
                    ax6.set_xlim(0, max_value * 1.15)
                    
                    # Add value labels
                    for i, v in enumerate(top_evening_employees.sort_values()):
                        ax6.text(v + (max_value * 0.02), i, f"${v:,.0f}", va='center', fontweight='bold')
                    
                    st.pyplot(fig6)

                    st.markdown(f"""
        **🔍 {shift_name} Performance:**  
        - **Purpose:** Identify shift-specific superstars  
        - **Key Metric:** Gross sales during {shift_name}  
        - **Scheduling Impact:** Match performers to optimal shifts  
        - **Pattern Recognition:** Reveal time-of-day strengths  
        
        *Pro Tip: Use this to create specialized shift teams*  
        """)

                else:
                    st.info(f"No data available for {evening_shift}")
            except Exception as e:
                st.error(f"Error generating {evening_shift} visualization: {str(e)}")

        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 8: Employee Shift History ===
        # Add this to your app after the Shift efficiency Analysis section:
        with st.expander("👥 Employee Shift History", expanded=False):
            shift_count_fig = plot_employee_shift_type_count(df)
            st.pyplot(shift_count_fig)    

            st.markdown("""
    **🔍 Shift Participation Analysis:**  
    - **Purpose:** Track employee availability patterns  
    - **Key Metric:** Number of shifts worked  
    - **Trend Insight:** Identify burnout risks  
    - **Coverage Planning:** Balance experience distribution  
    
    *Pro Tip: Green cells indicate frequent shift workers*  
    """)          

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 9: Employee 1st Shift Efficiency - Ranked ===
        with st.expander("🏆 Employee 1st Shift Efficiency - Ranked", expanded=False):
            morning_shift = st.session_state.shift_config['Shift 1']['name']
            morning_score_fig = plot_simplified_employee_morning_score(df)
            if morning_score_fig:
                st.pyplot(morning_score_fig)

                st.markdown(f"""
        **🔍 Efficiency Scoring System:**  
        - **Metrics Combined:**  
          1. Total sales  
          2. Average transaction value  
          3. Shifts worked  
          4. Sales per shift  
        - **Scoring:** 4-3-2-1 weighted ranking  
        - **Color Coding:** Darker greens = higher efficiency  
        
        *Pro Tip: Top 3 scorers get priority for premium shifts*  
        """)

            else:
                st.info("No morning shift data available for scoring")



        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === VISUALIZATION 10: Employee 2nd Shift Efficiency - Ranked ===
        with st.expander("🏆 Employee 2nd Shift Efficiency - Ranked", expanded=False):
            evening_shift = st.session_state.shift_config['Shift 2']['name']
            evening_score_fig = plot_simplified_employee_evening_score(df)
            if evening_score_fig:
                st.pyplot(evening_score_fig)

                st.markdown(f"""
        **🔍 Efficiency Scoring System:**  
        - **Metrics Combined:**  
          1. Total sales  
          2. Average transaction value  
          3. Shifts worked  
          4. Sales per shift  
        - **Scoring:** 4-3-2-1 weighted ranking  
        - **Color Coding:** Darker greens = higher efficiency  
        
        *Pro Tip: Top 3 scorers get priority for premium shifts*  
        """)
                
            else:
                st.info("No evening shift data available for scoring")

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === AI OPTIMIZED SCHEDULE (COLLAPSIBLE) ===
        with st.expander("🤖 AI Optimized Labor Schedule", expanded=True):
            # Get employee availability
            availability = get_employee_availability(df)
            
            # Generate and display schedule
            schedule_fig = plot_weekly_schedule_with_availability(df, availability)
            st.pyplot(schedule_fig)
        
        # Export options after schedule is generated
        st.subheader("📤 Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv_link = get_csv_download_link()
            if csv_link:
                if st.download_button(
                    label="📄 Download CSV",
                    data=st.session_state.schedule_df.to_csv().encode('utf-8'),
                    file_name="x_golf_schedule.csv",
                    mime="text/csv",
                    key="download-csv"
                ):
                    st.success("CSV file downloaded successfully!")
            else:
                st.button("📄 Download CSV", disabled=True, help="Generate schedule first")
        
        with export_col2:
            excel_link = get_excel_download_link()
            if excel_link:
                # For Excel, we need to create the file in memory
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.schedule_df.to_excel(writer, sheet_name='Schedule')
                    
                    # Format Excel
                    workbook = writer.book
                    worksheet = writer.sheets['Schedule']
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    for col_num, value in enumerate(st.session_state.schedule_df.columns.values):
                        worksheet.write(0, col_num + 1, value, header_format)
                
                buffer.seek(0)
                
                if st.download_button(
                    label="📊 Download Excel",
                    data=buffer,
                    file_name="x_golf_schedule.xlsx",
                    mime="application/vnd.ms-excel",
                    key="download-excel"
                ):
                    st.success("Excel file downloaded successfully!")
            else:
                st.button("📊 Download Excel", disabled=True, help="Generate schedule first")
            
        with export_col3:
            pdf_link = get_pdf_download_link()
            if pdf_link:
                # For PDF, we need to create the file in memory
                buffer = io.BytesIO()
                with PdfPages(buffer) as pdf:
                    pdf.savefig(st.session_state.schedule_fig)
                
                buffer.seek(0)
                
                if st.download_button(
                    label="📑 Download PDF",
                    data=buffer,
                    file_name="x_golf_schedule.pdf",
                    mime="application/pdf",
                    key="download-pdf"
                ):
                    st.success("PDF file downloaded successfully!")
            else:
                st.button("📑 Download PDF", disabled=True, help="Generate schedule first")
                
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.error(traceback.format_exc())