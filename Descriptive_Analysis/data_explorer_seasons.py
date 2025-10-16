import pandas as pd
import os
import datetime

location_base = os.path.dirname(os.path.abspath(__file__))
location_root = os.path.dirname(os.path.dirname((location_base)))
location_data = os.path.join(location_root, "outputs")

os.makedirs(location_data, exist_ok=True)

PRINT_OUTPUT = False

filepath_merged_cleaned_data = os.path.join(location_base, "combined_data.csv")
data_source = pd.read_csv(filepath_merged_cleaned_data) # AKA df

def split_data_on_rats(data_set, season):
    """
        This is the primary function for splitting the data up to compare bat behaviour with and without rats.
        Updating this function will change the output for all data.

        'rat-minutes' is a simple way of splitting the data using the dataset2.csv column of 'rat_minutes'. The
        assumption is that a value of 0 means there were no rats within the time period of that row. However,
        there exists data where there is a rat arrival time and rat departure time, but zero rat minutes.

        'rat-period' is calculated from data contained only within dataset1.csv. It is based on the bat arrival time,
        and compared with the rat arrival and departure times. 

    """

    valid_seasons = ("Winter", "Spring") # These are the only seasons in the data file
    if (season not in valid_seasons): raise Exception("Incorrect season name")

    data_headers = data_set.columns.tolist()
    
    with_rats = []
    without_rats = []
    inconclusive = []

    for _, row in data_set.iterrows():
        row_season = row['season']
        if (row_season != season): continue

        try: bat_arrival = get_date(row['start_time'])
        except: bat_arrival = get_date(row['sstart_time'])
        rat_arrival = get_date(row['rat_period_start'])
        rat_departure = get_date(row['rat_period_end'])

        # Check if there is a malformed date string
        if (None in (bat_arrival, rat_arrival, rat_departure)):
            inconclusive.append(row)
            continue
        
        # See if bat lands while rat is still there
        if (bat_arrival >= rat_arrival and bat_arrival < rat_departure):
            with_rats.append(row)
            continue
        
        # See if rat has left before bat arrives
        if (rat_departure <= bat_arrival):
            without_rats.append(row)
            continue
        
        # Check if bat has already gone to food before rat arrives
        bat_landing_to_food = row['bat_landing_to_food']
        try: bat_landing_to_food = int(bat_landing_to_food)
        except: bat_landing_to_food = None

        if not bat_landing_to_food is None:
            time_delta = datetime.timedelta(seconds=bat_landing_to_food)
            bat_at_food = bat_arrival + time_delta

            if (rat_arrival > bat_at_food): 
                without_rats.append(row)
                continue
        
        print(bat_arrival, rat_arrival, rat_departure, bat_landing_to_food)
        inconclusive.append(row)
    
    return {
        "with-rats": pd.DataFrame(with_rats, columns=data_headers),
        "no-rats": pd.DataFrame(without_rats, columns=data_headers),
        "inconclusive": pd.DataFrame(inconclusive, columns=data_headers),
        "meta-headers": data_headers,
    }

def get_date(date_string):
    try: 
        return datetime.datetime.strptime(date_string, '%d/%m/%Y %H:%M')
    except: 
        try:
            return datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except:
            return None

def bat_waits_for_rat_to_leave(data_set, season):
    
    split_data = split_data_on_rats(data_set, season)
    data = [
        ("With Rats", split_data["with-rats"]),
        ("No Rats", split_data["no-rats"]),
    ]

    data_headers = split_data["meta-headers"]
    return_values = {}

    if (PRINT_OUTPUT): print("\n-----Bat waits for rat to leave-----")

    for section in data:

        title = section[0]
        data_values = section[1]

        bat_waits = []
        bat_doesnt_wait = []
        bat_doesnt_approach = []
        inconclusive = []

        if (len(data_values) > 0):
            for _, row in data_values.iterrows():
                try: bat_arrival = get_date(row['start_time'])
                except: bat_arrival = get_date(row['sstart_time'])
                rat_arrival = get_date(row['rat_period_start'])
                rat_departure = get_date(row['rat_period_end'])

                bat_landing_to_food = row['bat_landing_to_food']
                try: bat_landing_to_food = int(bat_landing_to_food)
                except: bat_landing_to_food = None

                # Skip over any malformed date entries:
                if (None in (bat_arrival, rat_arrival, rat_departure)):
                    inconclusive.append(row)
                    continue

                if not bat_landing_to_food is None:
                    time_delta = datetime.timedelta(seconds=bat_landing_to_food)
                    bat_approach = bat_arrival + time_delta
                else:
                    bat_doesnt_approach.append(row)
                    continue
                
                if (bat_approach > rat_departure): bat_waits.append(row)
                else: bat_doesnt_wait.append(row)

        if (PRINT_OUTPUT):
            total_rows = len(data_values)
            print(f"\n\t{title} (total times) [{total_rows}]")
            if (total_rows == 0):
                print("\t\t(no data in set)")
            else:
                print(f"\t\tBat waits: {len(bat_waits)} [{len(bat_waits)/total_rows*100:.2f}%]")
                print(f"\t\tBat doesn't wait: {len(bat_doesnt_wait)} [{len(bat_doesnt_wait)/total_rows*100:.2f}%]")
                print(f"\t\tBat doesn't approach: {len(bat_doesnt_approach)} [{len(bat_doesnt_approach)/total_rows*100:.2f}%]")
                if (len(inconclusive) > 0): print(f"\t\tinconclusive: {len(inconclusive)} [{len(inconclusive)/total_rows*100:.2f}%]")
        
        return_values[title] = {
            "bat_waits": pd.DataFrame(bat_waits, columns=data_headers),
            "bat_doesnt_wait": pd.DataFrame(bat_doesnt_wait, columns=data_headers),
            "bat_doesnt_approach": pd.DataFrame(bat_doesnt_approach, columns=data_headers),
            "inconclusive": pd.DataFrame(inconclusive, columns=data_headers),
        }
    return return_values

def bat_sees_rat_as_competitor(data_set, season):
    
    if (PRINT_OUTPUT): print("\n-----Bat sees rat as competitor-----")

    split_data = split_data_on_rats(data_set, season)
    baseline_data = split_data["no-rats"]
    testing_data = split_data["with-rats"]
    
    ## Step 1, get a baseline:
    # Get all data where there are no rats
    # Get average time-to-food for bat
    # Get average risk value

    if (PRINT_OUTPUT): print(f"\n\tBaseline data (no rats) [{len(baseline_data)}]:")
    if (len(baseline_data) > 0):
        baseline_battofood_mean = baseline_data['bat_landing_to_food'].mean()
        baseline_battofood_mode = baseline_data['bat_landing_to_food'].mode()[0]
        baseline_risk_mean = baseline_data['risk'].mean()
        baseline_battofood_quant = baseline_data['bat_landing_to_food'].quantile([0.25, 0.50, 0.75, 1.00])
    
        if (PRINT_OUTPUT):
            print(f"\n\t\tBat risk: {baseline_risk_mean*100:.2f}%")
            print(f"\n\t\tBat to food: {baseline_battofood_mean:.2f}s (mean) | {float(baseline_battofood_mode)}s (mode)")
            for index, a_value in enumerate(baseline_battofood_quant, 1):
                print(f"\t\tQ{index}: {a_value}s")
    else:
        baseline_battofood_mean = 0
        baseline_battofood_mode = 0
        baseline_risk_mean = 0
        baseline_battofood_quant = [0,0,0,0]
        if (PRINT_OUTPUT): print(f"\t\t(no data in set)")

    ## Step 2, get relevant data set
    # Get all data where rats are present
    # Get average time-to-food for bat
    # Get average risk value

    if (PRINT_OUTPUT): print(f"\n\tHypothesis data (with rats) [{len(testing_data)}]:")
    if (len(testing_data) > 0):
        testing_battofood_mean = testing_data['bat_landing_to_food'].mean()
        testing_battofood_mode = testing_data['bat_landing_to_food'].mode()[0]
        testing_risk_mean = testing_data['risk'].mean()
        testing_battofood_quant = testing_data['bat_landing_to_food'].quantile([0.25, 0.50, 0.75, 1.00])

        if (PRINT_OUTPUT):
            print(f"\n\t\tBat risk: {testing_risk_mean*100:.2f}%")
            print(f"\n\t\tBat to food: {testing_battofood_mean:.2f}s (mean) | {testing_battofood_mode}s (mode)")
            for index, a_value in enumerate(testing_battofood_quant, 1):
                print(f"\t\tQ{index}: {a_value}s")

    else:
        testing_battofood_mean = 0
        testing_battofood_mode = 0
        testing_risk_mean = 0
        testing_battofood_quant = [0,0,0,0]
        if (PRINT_OUTPUT): print(f"\t\t(no data in set)")

    return {
        "no-rats":{
            "time-to-food-mean": baseline_battofood_mean,
            "time-to-food-mean": baseline_battofood_mode,
            "risk": baseline_risk_mean,
        },
        "with-rats": {
            "time-to-food-mean": testing_battofood_mean,
            "time-to-food-mean": testing_battofood_mode,
            "risk": testing_risk_mean,
        }
    }

def examine_behaviours(data_set, season):

    split_data = split_data_on_rats(data_set, season)
    data = [
        ("With Rats", split_data["with-rats"]),
        ("No Rats", split_data["no-rats"]),
    ]

    data_headers = split_data["meta-headers"]
    return_values = {}

    if (PRINT_OUTPUT): print("\n-----Behaviour breakdown-----")

    for section in data:

        title = section[0]
        data_values = section[1]

        if (len(data_values) == 0):
            if (PRINT_OUTPUT): print("\t(no data in set)")
            continue
	
        def rat_before_bat(data_entry):
            try: bat_arrival = get_date(data_entry['start_time'])
            except: bat_arrival = get_date(data_entry['sstart_time'])
            rat_arrival = get_date(data_entry['rat_period_start'])

            if (None in (bat_arrival, rat_arrival)): return None

            if (bat_arrival > rat_arrival): return True
            return False

        rat_already_there = []
        reward_without_risk = []
        risk_without_reward = []
        risk_with_reward = []
        inconclusive = []

        for _, an_entry in data_values.iterrows():
            rbb = rat_before_bat(an_entry)

            if (rbb is None): 
                inconclusive.append(rbb)
                continue

            if (rbb):
                rat_already_there.append(an_entry)
            
            if (an_entry["risk"] == 1):
                if (an_entry["reward"] == 1):
                    risk_with_reward.append(an_entry)
                elif (an_entry["reward"] == 0):
                    risk_without_reward.append(an_entry)
            elif (an_entry["reward"] == 1):
                reward_without_risk.append(an_entry)

        summary = {
            "rat_already_there": pd.DataFrame(rat_already_there, columns=data_headers),
            "reward_without_risk": pd.DataFrame(reward_without_risk, columns=data_headers),
            "risk_without_reward": pd.DataFrame(risk_without_reward, columns=data_headers),
            "risk_with_reward": pd.DataFrame(risk_with_reward, columns=data_headers),
            "inconclusive": pd.DataFrame(inconclusive, columns=data_headers),
        }
        if (PRINT_OUTPUT):
            total_rows = len(data_values)
            print(f"\n\t{title} [{total_rows}]")
            print(f"\t\tRat present before bat: {len(rat_already_there)} [{len(rat_already_there)/total_rows*100:.2f}%]")
            print(f"\t\tReward without risk: {len(reward_without_risk)} [{len(reward_without_risk)/total_rows*100:.2f}%]")
            print(f"\t\tRisk without reward: {len(risk_without_reward)} [{len(risk_without_reward)/total_rows*100:.2f}%]")
            print(f"\t\tRisk with reward: {len(risk_with_reward)} [{len(risk_with_reward)/total_rows*100:.2f}%]")
            if (len(inconclusive) > 0): print(f"\t\tInconclusive: {len(inconclusive)} [{len(inconclusive)/total_rows*100:.2f}%]")

        return_values[title] = summary

    return return_values

def seasonal_breakdown(season):
    split_data = split_data_on_rats(data_source, season)
    if (PRINT_OUTPUT):
        print(f"-----Basic Split ({season})-----")
        print(f"\n\tTotal no-rats: {len(split_data['no-rats'])}")
        print(f"\tTotal with-rats: {len(split_data['with-rats'])}")
        if (len(split_data['inconclusive']) > 0): print(f"\tTotal inconclusive: {len(split_data['inconclusive'])}")

    # Then call the functions that break down the summaries:

    #bat_rat_competitor = bat_sees_rat_as_competitor(data_source, season)
    bat_wait_for_rat = bat_waits_for_rat_to_leave(data_source, season)
    bat_behaviours = examine_behaviours(data_source, season)

    return {
        #'bat_rat_competitor': bat_rat_competitor,
        'rat_norat_split': split_data,
        'bat_wait_for_rat': bat_wait_for_rat,
        'bat_behaviours': bat_behaviours,
    }

def export_to_json(dict_object, filename="seasonal_data_dump.json"):
    import json

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pd.Series):
                return obj.tolist()  # Or obj.to_json(orient='values')
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return json.JSONEncoder.default(self, obj)
    
    output_path = os.path.join(location_data, filename)
    print(f"Outputting test file to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(dict_object, f, indent=4, cls=CustomEncoder)

def get_seasonal_breakdown(export=False):
    seasonal_data = {
        "Winter": seasonal_breakdown("Winter"),
        "Spring": seasonal_breakdown("Spring"),
    }
    if (export):
        export_to_json(seasonal_data)
    
    return seasonal_data

#seasonal_data = get_seasonal_breakdown()
#df__winter = seasonal_data["Winter"]["bat_wait_for_rat"]["No Rats"]["bat_waits"]
#df__spring = seasonal_data["Spring"]["bat_wait_for_rat"]["No Rats"]["bat_waits"]