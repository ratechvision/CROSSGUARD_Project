# convert_threat_reports.py
import json
import os

# Define input and output directories
input_dirs = ["F:/PROJECTS2025/UnifiedCyberThreatDetection/data/threat_reports/campaign",
              "F:/PROJECTS2025/UnifiedCyberThreatDetection/data/threat_reports/intrusion-set"]
output_dir = "F:/PROJECTS2025/UnifiedCyberThreatDetection/data/threat_reports/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Counter for new threat report IDs
counter = 1

# Process each input directory
for input_dir in input_dirs:
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as f:
                data = json.load(f)

                # Create new threat report
                new_report = {
                    "id": f"TR{counter:03d}",
                    "date": "2025-03-26",
                    "title": data.get("name", "Unknown Threat"),
                    "description": data.get("description", ""),
                    "threat_level": "High",
                    "label": "Threat"
                }

                # Save the new report
                output_file = os.path.join(output_dir, f"threat_report_{counter}.json")
                with open(output_file, 'w') as f:
                    json.dump(new_report, f, indent=4)

                counter += 1

print(f"Converted {counter - 1} threat reports to {output_dir}")