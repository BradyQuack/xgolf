# AI Shift Optimizer

## Overview
AI Shift Optimizer is a Streamlit application designed to help businesses optimize their employee scheduling by analyzing historical sales data, employee performance, and availability to create data-driven shift schedules. The application uses machine learning techniques to identify the most efficient staff assignments while balancing workload and maximizing revenue potential.

## Features

### Data Analysis
- Sales heatmap by weekday and hour to identify peak revenue periods
- Shift-specific revenue analysis to validate shift timing effectiveness
- Employee performance tracking for total sales, average transaction value, and shift efficiency
- Customizable visualizations with adjustable employee count displays

### Role Configuration
- Create and customize multiple role types (Bartender, Server, etc.)
- Toggle optimization mode per role:
  - Optimized: AI prioritizes top performers
  - Non-optimized: Equal shift distribution
- Staff requirement configuration for each role

### Shift Management
- Flexible shift definition with customizable start/end times
- Role-specific staffing needs per shift
- Real-time shift overlap detection
- Automated addition/removal of shifts

### Employee Availability Management
- Track individual availability preferences:
  - Maximum shifts per week
  - Preferred days and shifts
  - Role capabilities and preferences
- Performance-based scheduling prioritization
- Automatic conflict detection

### AI-Optimized Scheduling
- Multi-dimensional optimization considering:
  - Employee performance metrics
  - Shift revenue patterns
  - Employee availability and preferences
  - Role requirements
  - Fair distribution of shifts
- Visual schedule presentation with efficiency scores

### Export Options
- Download optimized schedules in CSV format
- Export to Excel with formatting
- Save schedule as PDF for easy sharing

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-shift-optimizer.git
cd ai-shift-optimizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Dependencies
The application requires the following Python packages:
```
streamlit
pandas
matplotlib
seaborn
numpy
openpyxl
xlsxwriter
```

## Usage

### Running the Application
1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open the application in your web browser (typically http://localhost:8501)

### Data Requirements
The CSV file must contain the following columns:
```
Date (format: MM/DD/YYYY)
Time (format: HH:MM:SS)
Gross Sales (currency value)
Employee (employee names)
```

### Workflow
1. **Upload Data**: Start by uploading your CSV file with sales data
2. **Configure Roles**: Set up different job roles in the sidebar
3. **Configure Shifts**: Define shift timings and staffing requirements
4. **Set Employee Availability**: Configure each employee's availability
5. **View Analytics**: Explore the various performance visualizations
6. **Generate Schedule**: The AI-optimized schedule is automatically created
7. **Export Results**: Download the optimized schedule in your preferred format

## Customization

### Configuration Options
- Adjust the number of employees displayed in visualizations using the sliders
- Customize shift times to match your business hours
- Configure role-specific staffing requirements
- Set employee availability preferences

## Troubleshooting
- If visualizations appear empty, check if your data matches the required format
- Ensure all shifts have appropriate time ranges that don't overlap incorrectly
- Verify that employee names in the data exactly match availability settings
