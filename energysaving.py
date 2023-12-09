import pandas as pd
from datetime import timedelta

# Load the Excel file
file_path = 'energy_data_merged.xlsx'
data = pd.read_excel(file_path)

# Convert 'timestamp' to datetime and extract date
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['date'] = data['timestamp'].dt.date

# Function to calculate AC status based on control period
from datetime import timedelta


def calculate_ac_status(data, control_period_minutes):
    # Convert control period to the number of data points in the period
    # Assuming data points are spaced evenly (e.g., every 10 seconds)
    # Assuming data points are spaced evenly (e.g., every 10 seconds)
    period_length = int(control_period_minutes * 60 / 10)  # Change 10 to your actual data spacing in seconds

    # Initialize the AC status list
    ac_status = [0] * len(data)

    # Initialize a variable to keep track of when the AC was last turned on
    last_on_time = None

    # Iterate over the data points
    for i in range(len(data)):
        # Check if there was occupancy in the current segment
        if data['new_occupied'].iloc[i] == 1:
            last_on_time = i

        # Check if the current segment is within control_period_minutes since the AC was last turned on
        if last_on_time is not None and (i - last_on_time) < period_length:
            ac_status[i] = 1

    # Assign the AC status to the data frame
    data['ac_status'] = ac_status
    return data



# Function to apply revised basic AC control strategy
def revised_basic_ac_control_strategy(data,minutes):
    delay = pd.Timedelta(minutes=minutes)
    data['revised_ac_status'] = 0  # Initialize all values to 0 (AC off)

    for date in data['date'].unique():
        # Filter data for each day
        daily_data = data[data['date'] == date]

        # Turn on AC when the first 'new_occupied' value of 1 is encountered
        first_occupied_time = daily_data[daily_data['new_occupied'] == 1]['timestamp'].min()
        if pd.notna(first_occupied_time):
            on_time = first_occupied_time
            on_index = data[data['timestamp'] >= on_time].index.min()
            if pd.notna(on_index):
                data.loc[on_index:, 'revised_ac_status'] = 1

        # Turn off AC after the last 'new_occupied' value of 1 for the day plus a delay
        last_occupied_time = daily_data[daily_data['new_occupied'] == 1]['timestamp'].max()
        if pd.notna(last_occupied_time):
            off_time = last_occupied_time
            off_index = data[data['timestamp'] >= off_time].index.min()
            if pd.notna(off_index):
                data.loc[off_index:, 'revised_ac_status'] = 0

    return data

# Apply revised basic AC control strategy


# Iterate over control periods from 5 to 30 minutes and calculate energy saving
energy_saving_rate=[]
for minutes in range(10, 81, 10):
    # Apply control period AC status calculation
    cdata = revised_basic_ac_control_strategy(data, minutes)
    data_with_ac_status = calculate_ac_status(cdata.copy(), minutes)
    data_with_ac_status['ac_status']=data_with_ac_status['ac_status']*data_with_ac_status['revised_ac_status']

    # Calculate proportion of AC on time
    ac_on_proportion_control = data_with_ac_status['ac_status'].mean()
    ac_on_proportion_revised = data_with_ac_status['revised_ac_status'].mean()

    # Calculate energy saving
    energy_saving = 1 - (ac_on_proportion_control / ac_on_proportion_revised)
    energy_saving_rate.append(energy_saving)
    print(f"Control Period: {minutes} minutes - Energy Saving: {energy_saving * 100:.2f}%")


import pandas as pd
import matplotlib.pyplot as plt

# Since I cannot access the actual data file, I will simulate the energy saving calculations for the plot
# This is a placeholder for actual calculation logic
control_periods = list(range(10, 81, 10))


# Now let's create a scatter plot with a line connecting the dots, similar to the provided example
plt.figure(figsize=(10, 5))
plt.plot(control_periods, energy_saving_rate, marker='o', linestyle='-', color='blue', label='Energy saving time')
plt.title('Energy Savings vs. Control Period')
plt.xlabel('Control Period (minutes)')
plt.ylabel('Energy Saving (%)')
plt.xticks(control_periods)
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.dates as mdates

# Assuming data has already been processed and is in the variable updated_data_with_basic_ac
# Filtering the data for the specific date 2019-11-27
data_nov_27 = data_with_ac_status[data_with_ac_status['date'] == pd.to_datetime('2019-11-27').date()]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Air Conditioning Status based on control strategy

ff=22
# People Presence based on 'new_occupied' column
axs[0].plot(data_nov_27['timestamp'], data_nov_27['new_occupied'], marker='o', linestyle='-', color='green', markersize=2, label='Occupancy Presence')
axs[0].set_title('Occupancy Presence',fontsize=ff)
axs[0].set_ylabel('Presence (0 or 1)',fontsize=ff)
axs[0].legend(fontsize=ff)
axs[0].grid(True)

# Basic Air Conditioning Status
axs[1].plot(data_nov_27['timestamp'], data_nov_27['revised_ac_status'], marker='o', linestyle='-', color='red', markersize=2, label='Basic AC Status')
axs[1].set_title('Basic Air Conditioning Status',fontsize=ff)
# axs[1].set_xlabel('Time',fontsize=ff)
axs[1].set_ylabel('Basic AC Status',fontsize=ff)
axs[1].legend(fontsize=ff)
axs[1].grid(True)

axs[2].plot(data_nov_27['timestamp'], data_nov_27['ac_status'], marker='o', linestyle='-', color='blue', markersize=2, label='OCC AC Control Status')
axs[2].set_title('OCC Air Conditioning Status',fontsize=ff)
axs[2].set_ylabel('OCC AC Status',fontsize=ff)
axs[2].legend(loc='upper left',fontsize=ff)
axs[2].grid(True)

# Formatting the x-axis to show hour:minute format
axs[2].xaxis.set_major_locator(mdates.HourLocator(interval=1))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
