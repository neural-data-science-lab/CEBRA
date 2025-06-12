#!/usr/bin/env python3
"""
Context-Aware Token Labeling using Anthropic API
Processes tokens with semantic context and generates probabilistic labels
"""

import pandas as pd
import json
import time
import os
from typing import List, Dict, Any
from anthropic import Anthropic
import argparse
from datetime import datetime
from dotenv import load_dotenv

class TokenLabeler:
    def __init__(self, api_key: str, context_window: int = 10):
        """
        Initialize the token labeler

        Args:
            api_key: Anthropic API key
            context_window: Number of tokens before/after to include as context
        """
        load_dotenv("./.env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)
        self.context_window = context_window
        self.processed_count = 0

    def clean_tokens(self, df: pd.DataFrame) -> List[str]:
        """Clean and extract tokens from the CSV DataFrame"""
        print("Cleaning tokens...")

        # Get the first column (assuming it contains the tokens)
        column_name = df.columns[0]
        raw_tokens = df[column_name].tolist()

        # Clean tokens - remove the [""] wrapper
        clean_tokens = []
        for token in raw_tokens:
            if pd.isna(token) or token == '':
                continue

            # Remove [" and "] wrapper and handle escaped quotes
            cleaned = str(token).strip()
            if cleaned.startswith('["') and cleaned.endswith('"]'):
                cleaned = cleaned[2:-2]  # Remove [" and "]
                cleaned = cleaned.replace('""', '"')  # Handle escaped quotes

            if cleaned.strip():
                clean_tokens.append(cleaned.strip())

        print(f"Cleaned {len(clean_tokens)} tokens")
        return clean_tokens

    def get_context(self, tokens: List[str], index: int) -> Dict[str, Any]:
        """Get context around a specific token"""
        start_idx = max(0, index - self.context_window)
        end_idx = min(len(tokens), index + self.context_window + 1)

        context_tokens = tokens[start_idx:end_idx]
        target_token = tokens[index]

        # Create context string
        context_before = tokens[start_idx:index] if index > start_idx else []
        context_after = tokens[index + 1:end_idx] if index + 1 < end_idx else []

        return {
            'target_token': target_token,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': ' '.join(context_tokens)
        }

    def create_labeling_prompt(self, context_info: Dict[str, Any]) -> str:
        """Create a prompt for the API to label a token"""
        return f"""You are a semantic token labeling expert. Analyze the given token within its context and provide labels for these four categories:

1. **location** - Physical places, geographical locations, spatial references
2. **characters** - People, character names, pronouns referring to people, roles
3. **emotions** - Emotional states, feelings, mood descriptors
4. **time** - Temporal references, time periods, time-related words

**Target Token:** "{context_info['target_token']}"

**Context Before:** {' '.join(context_info['context_before'][-5:]) if context_info['context_before'] else '[none]'}
**Context After:** {' '.join(context_info['context_after'][:5]) if context_info['context_after'] else '[none]'}

For each category, provide:
- **value**: The specific semantic label (e.g., "office", "Carmen Reed", "tired", "afternoon") or "null" if not applicable
- **confidence**: Float between 0.0-1.0 indicating your confidence in the label

Consider the context when labeling. If the token doesn't directly contain a category but the context suggests it should be labeled (e.g., pronouns referring to characters, implicit time/location), include those labels.

Respond in this exact JSON format:
{{
  "location": {{"value": "label_or_null", "confidence": 0.0}},
  "characters": {{"value": "label_or_null", "confidence": 0.0}},
  "emotions": {{"value": "label_or_null", "confidence": 0.0}},
  "time": {{"value": "label_or_null", "confidence": 0.0}}
}}"""

    def label_token(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Send a token to the API for labeling"""
        prompt = self.create_labeling_prompt(context_info)

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Use Haiku for cost efficiency
                max_tokens=300,
                temperature=0.1,  # Low temperature for consistent formatting
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract and parse JSON response
            response_text = response.content[0].text.strip()

            # Try to extract JSON from response
            try:
                # Look for JSON block
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    raise ValueError("No JSON found in response")

                labels = json.loads(json_text)

                # Validate structure
                required_categories = ['location', 'characters', 'emotions', 'time']
                for category in required_categories:
                    if category not in labels:
                        labels[category] = {"value": "null", "confidence": 0.9}
                    elif 'value' not in labels[category] or 'confidence' not in labels[category]:
                        labels[category] = {"value": "null", "confidence": 0.9}

                return labels

            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response_text}")
                # Return default structure on parsing error
                return {
                    "location": {"value": "null", "confidence": 0.9},
                    "characters": {"value": "null", "confidence": 0.9},
                    "emotions": {"value": "null", "confidence": 0.9},
                    "time": {"value": "null", "confidence": 0.9}
                }

        except Exception as e:
            print(f"API error for token '{context_info['target_token']}': {e}")
            # Return default structure on API error
            return {
                "location": {"value": "null", "confidence": 0.9},
                "characters": {"value": "null", "confidence": 0.9},
                "emotions": {"value": "null", "confidence": 0.9},
                "time": {"value": "null", "confidence": 0.9}
            }

    def process_tokens(self, tokens: List[str], batch_size: int = 1, delay: float = 0.5) -> List[Dict[str, Any]]:
        """Process all tokens with API labeling"""
        results = []
        total_tokens = len(tokens)

        print(f"Processing {total_tokens} tokens...")
        print(f"Using context window of {self.context_window} tokens")
        print(f"API delay: {delay} seconds between calls")

        for i, token in enumerate(tokens):
            # Get context for this token
            context_info = self.get_context(tokens, i)

            # Get labels from API
            labels = self.label_token(context_info)

            # Create result entry
            result = {
                "token": token,
                "index": i,
                "labels": labels
            }

            results.append(result)
            self.processed_count += 1

            # Progress reporting
            if (i + 1) % 10 == 0 or i + 1 == total_tokens:
                print(f"Processed {i + 1}/{total_tokens} tokens ({((i + 1)/total_tokens)*100:.1f}%)")

            # Rate limiting
            if i < total_tokens - 1:  # Don't delay after the last token
                time.sleep(delay)

        return results

    def generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics from the results"""
        stats = {
            'location': {'count': 0, 'unique_values': set()},
            'characters': {'count': 0, 'unique_values': set()},
            'emotions': {'count': 0, 'unique_values': set()},
            'time': {'count': 0, 'unique_values': set()}
        }

        for result in results:
            for category in stats.keys():
                value = result['labels'][category]['value']
                if value != "null":
                    stats[category]['count'] += 1
                    stats[category]['unique_values'].add(value)

        # Convert sets to lists and counts
        final_stats = {}
        for category, data in stats.items():
            final_stats[category] = {
                'count': data['count'],
                'unique_count': len(data['unique_values']),
                'unique_values': list(data['unique_values'])
            }

        return final_stats

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file"""
        stats = self.generate_statistics(results)

        final_output = {
            "metadata": {
                "total_tokens": len(results),
                "categories": ["location", "characters", "emotions", "time"],
                "processing_date": datetime.now().isoformat(),
                "description": "Context-aware semantic token labeling using Anthropic API with probabilistic label generation",
                "context_window": self.context_window,
                "statistics": stats
            },
            "tokens": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")
        print(f"Total tokens processed: {len(results)}")
        print("\nStatistics:")
        for category, data in stats.items():
            percentage = (data['count'] / len(results)) * 100
            print(f"- {category.capitalize()}: {data['count']} tokens ({percentage:.1f}%), {data['unique_count']} unique values")


def main():
    parser = argparse.ArgumentParser(description='Label tokens using Anthropic API')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--api-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--context-window', type=int, default=10, help='Context window size (default: 10)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls in seconds (default: 0.5)')

    args = parser.parse_args()

    # Get API key
    load_dotenv("./.env")
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please provide API key via --api-key argument or ANTHROPIC_API_KEY environment variable")
        return

    try:
        # Read CSV file
        print(f"Reading {args.input_file}...")
        df = pd.read_csv(args.input_file)

        # Initialize labeler
        labeler = TokenLabeler(api_key, context_window=args.context_window)

        # Clean tokens
        tokens = labeler.clean_tokens(df)

        if not tokens:
            print("Error: No valid tokens found in the input file")
            return

        # Process tokens
        results = labeler.process_tokens(tokens, delay=args.delay)

        # Save results
        labeler.save_results(results, args.output_file)

        print(f"\nProcessing complete!")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()


# Example usage:
# python token_labeler.py tokens.csv output.json --api-key your_api_key_here
#
# Or with environment variable:
# export ANTHROPIC_API_KEY=your_api_key_here
# python token_labeler.py tokens.csv output.json --context-window 15 --delay 0.3