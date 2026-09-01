import json
import csv
import os

# Set your data folder path
data_folder = 'F:/PROJECTS2025/UnifiedCyberThreatDetection/data/threat_reports'

# Create output directory for CSV files if it doesn't exist
output_folder = os.path.join(data_folder, 'csv_output')
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file in the directory
for filename in os.listdir(data_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(data_folder, filename)
        csv_path = os.path.join(output_folder, filename.replace('.json', '.csv'))
        
        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                # Assuming the JSON data is a list of dictionaries with consistent keys
                if data and isinstance(data, list):
                    writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                elif isinstance(data, dict):
                    # If it's a single dictionary, write as one row
                    writer = csv.DictWriter(csv_file, fieldnames=data.keys())
                    writer.writeheader()
                    writer.writerow(data)
                
            print(f"Successfully converted {filename} to CSV")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("Conversion complete!")