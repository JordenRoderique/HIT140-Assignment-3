import os
import csv
import pandas as pd
import datetime

location_base = os.path.dirname(os.path.abspath(__file__))
location_root = os.path.dirname(os.path.dirname((location_base)))
location_data = os.path.join(location_root, "data")

output_file = os.path.join(location_base, "combined_data.csv")
print(f"Export location set to: {output_file}")

def get_date(date_string):
    try: 
        return datetime.datetime.strptime(date_string, '%d/%m/%Y %H:%M')
    except: 
        try:
            return datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except:
            return None

def read_in_data(data_path):

    with open(data_path, mode='r') as file:
        reader = csv.reader(file)
        header_data = None
        return_data = []
        
        for a_row in reader:
            
            if (header_data is None):
                header_data = a_row
            else:
                entry_dict = {}
                for index, a_datum in enumerate(a_row):
                    
                    data_header = header_data[index]

                    potential_date = get_date(a_datum)
                    if (potential_date): a_datum = potential_date
                    
                    entry_dict[data_header] = a_datum
                return_data.append(entry_dict)
        
    return header_data, return_data

def clean_csv_data(csv_number, row_data):
    def enforce_numeric(data_entry, placeholder="NaN"):
        try:
            data_entry = float(data_entry)
            return data_entry
        except: return placeholder
    
    def invalidate_numeric(data_entry, min=None, max=None, placeholder="NaN"):
        try: data_entry = float(data_entry)
        except: return placeholder

        if ((not min is None) and (data_entry < min)): return placeholder
        
        if ((not max is None) and (data_entry > max)): return placeholder

        return data_entry

    def clamp_numeric(data_entry, min=None, max=None, placeholder="NaN"):
        try: data_entry = float(data_entry)
        except: return placeholder
        
        if ((not min is None) and (data_entry < min)): data_entry = min
        if ((not max is None) and (data_entry > max)): data_entry = max

        return data_entry

    number_headers = []

    if (csv_number == 1):
        number_headers = ["seconds_after_rat_arrival", "hours_after_sunset"]
        
        # latency hygiene constraint copied from clean_dataset1.py
        row_data["bat_landing_to_food"] = invalidate_numeric(row_data["bat_landing_to_food"], 1, None)

        if (str(row_data["season"]) == "0"): row_data["season"] = "Winter"
        elif (str(row_data["season"]) == "1"): row_data["season"] = "Spring"
        else: row_data["season"] = None
    
    elif (csv_number == 2):
        number_headers = ["month", "hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]

    for a_header in number_headers: row_data[a_header] = enforce_numeric(row_data[a_header])

    return row_data

def comine_data_sources():
    def match_30min_chunk(start_time, data_set):
        
        time_delta = datetime.timedelta(minutes=30)

        for _, row in data_set.iterrows():
            chunk_start = get_date(row["time"])
            if (chunk_start is None): continue

            chunk_end = chunk_start + time_delta

            if (start_time >= chunk_start and start_time <= chunk_end):
                return row

        return None

    data_set_1 = os.path.join(location_data, "dataset1.csv")
    data_set_2 = os.path.join(location_data, "dataset2.csv")

    data_1_content = pd.read_csv(data_set_1)
    data_2_content = pd.read_csv(data_set_2)

    data_1_headers = data_1_content.columns.tolist()
    data_2_headers = data_2_content.columns.tolist()

    output_headers = data_1_headers + data_2_headers + ["csv_row_index", 'sort_date'] # Add row index for verifying end results (spot check)
    output_rows = []

    for index, row in data_1_content.iterrows():

        # Original data has 'start_time' as the column name, but somewhere it got changed to 'sstart_time'.
        # Adding backup so either version will work:
        try: s_time = get_date(row["start_time"])
        except: s_time = get_date(row["sstart_time"])

        print(f"Working on chunk {index+1}...")
        date_list = [
            s_time,
            get_date(row["rat_period_start"]),
            get_date(row["rat_period_end"])
        ]

        earliest_date = min(x for x in date_list if x is not None)

        chunk = match_30min_chunk(earliest_date, data_2_content)
        if (chunk is None): 
            print(f"\tCould not find match for row {index+1}")
            continue

        # Get a simple date string of the ealiest date for plotting
        sort_date = earliest_date.strftime("%Y-%m-%d")

        # Clean the variables:
        cleaned_row = clean_csv_data(1, row)
        cleaned_chunk = clean_csv_data(2, chunk)

        # Add all the data together as a list and add the index on the end (+2 for header and Excel numbering starts at 1):
        output_rows.append(cleaned_row.tolist() + cleaned_chunk.tolist() + [index+2, sort_date])
    
    while True:
        try:
            with open(output_file, 'w', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)

                # Write the header row
                csv_writer.writerow(output_headers)

                # Write the data rows
                csv_writer.writerows(output_rows)
            break
        except:
            print("File failed to write... please make sure the CSV file is not open.")
            input("Press enter to try again...")

comine_data_sources()