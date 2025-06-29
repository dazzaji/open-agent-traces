# python module6.py

import json
import os
import re

def clean_markdown_content(item_title, item_description):
    """
    Clean up markdown content to remove duplicate headers and unnecessary formatting
    while preserving the important content.
    """
    # Skip the first header that matches or contains the item title
    lines = item_description.split('\n')
    cleaned_lines = []
    
    # Skip headers that match the item title and "Expanded Item" text
    skip_line = False
    for i, line in enumerate(lines):
        # Skip lines with headers that duplicate or contain the item title
        if (line.startswith('#') and item_title.lower() in line.lower()) or "expanded item:" in line.lower():
            skip_line = True
            continue
        
        # Skip lines with "overview", "step", and header markers from the original markdown
        if re.match(r'^#{3,} (Step \d+:|Overview)', line):
            skip_line = True
            continue
            
        # Don't skip the rest of the content
        skip_line = False
        
        # Add the line to our cleaned lines
        cleaned_lines.append(line)
    
    # Remove redundant blank lines (more than 2 in a row)
    result_lines = []
    last_was_blank = False
    for line in cleaned_lines:
        if not line.strip():
            if last_was_blank:
                continue
            last_was_blank = True
        else:
            last_was_blank = False
        result_lines.append(line)
    
    return '\n'.join(result_lines)

def main():
    # Determine the directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    # Input file from Module 5 and output file for Module 6
    input_file = os.path.join(data_dir, "module5_output.json")
    output_file = os.path.join(data_dir, "revised_plan.md")
    
    # Load the Module 5 JSON data
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
    
    # Extract the key components
    goal = data.get("goal", "No goal provided.")
    selected_criteria = data.get("selected_criteria", [])
    revised_outline = data.get("revised_outline", {})
    
    # Build the Markdown content
    lines = []
    
    # Title 
    plan_title = revised_outline.get("plan_title", "No Plan Title")
    lines.append("# " + plan_title)
    lines.append("")
    
    # Goal
    lines.append("## Goal")
    lines.append("")
    lines.append(goal)
    lines.append("")
    
    # Success Criteria
    lines.append("## Success Criteria")
    lines.append("")
    if not selected_criteria:
        lines.append("No success criteria provided.")
    else:
        for criterion in selected_criteria:
            criteria_text = criterion.get("criteria", "")
            if criteria_text:
                lines.append(f"- **{criteria_text}**")
    lines.append("")
    
    # Plan overview
    plan_description = revised_outline.get("plan_description", "")
    if plan_description:
        lines.append("## Plan Overview")
        lines.append("")
        lines.append(plan_description)
        lines.append("")
    
    # Detailed steps
    lines.append("## Detailed Implementation Steps")
    lines.append("")
    
    plan_items = revised_outline.get("plan_items", [])
    for i, item in enumerate(plan_items, 1):
        item_title = item.get("item_title", f"Step {i}")
        item_description = item.get("item_description", "")
        
        # Add the step title
        lines.append(f"### Step {i}: {item_title}")
        lines.append("")
        
        # Clean up the description
        cleaned_content = clean_markdown_content(item_title, item_description)
        lines.append(cleaned_content)
        lines.append("")
    
    # Add closing note
    lines.append("---")
    lines.append("")
    lines.append("*This plan was generated using Dazza Greenwood's Agento framework.*")
    
    content = "\n".join(lines)
    
    # Write the formatted content to the output file
    try:
        with open(output_file, "w") as f:
            f.write(content)
        print(f"Module 6 completed. Revised plan exported to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    main()