"""
Load and process historical flood event databases.

This module provides functions to:
- Load Dartmouth Flood Observatory (DFO) flood events
- Filter events for Kyrgyzstan
- Merge flood labels with simulation data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_kyrgyzstan_floods(dfo_csv_path: str = 'data/FloodArchive.csv') -> pd.DataFrame:
    """
    Load Dartmouth Flood Observatory data for Kyrgyzstan.

    Args:
        dfo_csv_path: Path to DFO FloodArchive.csv
            Download from: https://floodobservatory.colorado.edu/Archives/

    Returns:
        DataFrame with columns:
            - start_date: Flood start date
            - end_date: Flood end date
            - Country: Country/region description
            - severity: moderate/high/critical
            - Dead: Number of deaths
            - Displaced: Number of displaced people
            - MainCause: Cause of flood

    Raises:
        FileNotFoundError: If the flood database file doesn't exist
    """
    # Check if file exists
    csv_path = Path(dfo_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Flood database not found at {dfo_csv_path}\n\n"
            f"Please download it from:\n"
            f"https://floodobservatory.colorado.edu/Archives/index.html\n\n"
            f"Steps:\n"
            f"1. Visit the URL above\n"
            f"2. Download 'FloodArchive.csv' or 'FloodArchive.xlsx'\n"
            f"3. Save it to: {csv_path.absolute()}"
        )

    logger.info(f"Loading flood database from {dfo_csv_path}")

    # Load data
    try:
        dfo = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        # Try alternative encodings
        dfo = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)

    logger.info(f"Loaded {len(dfo)} total flood events from database")

    # Filter for Kyrgyzstan
    kg_floods = dfo[
        (dfo['Country'].str.contains('Kyrgyzstan', case=False, na=False)) |
        (dfo['Country'].str.contains('Kyrgyz', case=False, na=False)) |
        (dfo['Country'].str.contains('Kirgiz', case=False, na=False))  # Alternative spelling
    ].copy()

    logger.info(f"Found {len(kg_floods)} flood events for Kyrgyzstan")

    if len(kg_floods) == 0:
        logger.warning("No flood events found for Kyrgyzstan in database")
        return pd.DataFrame(columns=[
            'start_date', 'end_date', 'Country', 'severity',
            'Dead', 'Displaced', 'MainCause'
        ])

    # Parse dates
    kg_floods['start_date'] = pd.to_datetime(kg_floods['Began'], errors='coerce')
    kg_floods['end_date'] = pd.to_datetime(kg_floods['Ended'], errors='coerce')

    # Map magnitude to severity
    magnitude_map = {
        '1': 'moderate',
        '1.0': 'moderate',
        '1.5': 'high',
        '2': 'critical',
        '2.0': 'critical'
    }
    kg_floods['severity'] = kg_floods['Magnitude'].astype(str).map(magnitude_map)
    kg_floods['severity'] = kg_floods['severity'].fillna('moderate')

    # Clean up columns
    kg_floods = kg_floods[[
        'start_date', 'end_date', 'Country',
        'severity', 'Dead', 'Displaced', 'MainCause'
    ]].dropna(subset=['start_date'])

    # Sort by date
    kg_floods = kg_floods.sort_values('start_date').reset_index(drop=True)

    # Log summary
    logger.info(f"Processed flood events:")
    logger.info(f"  - Date range: {kg_floods['start_date'].min()} to {kg_floods['start_date'].max()}")
    logger.info(f"  - Severity counts: {kg_floods['severity'].value_counts().to_dict()}")

    return kg_floods


def merge_flood_labels(simulation_df: pd.DataFrame,
                       flood_events: pd.DataFrame,
                       buffer_hours: int = 0) -> pd.DataFrame:
    """
    Merge simulation data with real flood event labels.

    Args:
        simulation_df: Hourly simulation results with 'timestamp' column
        flood_events: Historical flood events from load_kyrgyzstan_floods()
        buffer_hours: Hours to extend flood period (before/after)
            Use this to account for uncertainty in flood timing

    Returns:
        Simulation DataFrame with added columns:
            - flood: Binary indicator (0/1)
            - severity: Flood severity ('none', 'moderate', 'high', 'critical')
    """
    logger.info(f"Merging {len(flood_events)} flood events with simulation data")

    # Initialize columns
    simulation_df['flood'] = 0
    simulation_df['severity'] = 'none'

    if len(flood_events) == 0:
        logger.warning("No flood events to merge")
        return simulation_df

    # Label each flood event
    flood_hours_labeled = 0
    for idx, event in flood_events.iterrows():
        start = event['start_date']
        end = event.get('end_date', start + pd.Timedelta(days=1))
        severity = event.get('severity', 'moderate')

        # Apply buffer
        if buffer_hours > 0:
            start = start - pd.Timedelta(hours=buffer_hours)
            end = end + pd.Timedelta(hours=buffer_hours)

        # Find overlapping time period
        mask = (simulation_df['timestamp'] >= start) & \
               (simulation_df['timestamp'] <= end)

        hours_in_event = mask.sum()
        flood_hours_labeled += hours_in_event

        # Mark as flood
        simulation_df.loc[mask, 'flood'] = 1
        simulation_df.loc[mask, 'severity'] = severity

        logger.debug(f"Event {idx+1}: {start} to {end} ({severity}) - {hours_in_event} hours")

    total_hours = len(simulation_df)
    flood_percentage = (flood_hours_labeled / total_hours) * 100

    logger.info(f"Labeled {flood_hours_labeled} hours as flood events " +
                f"({flood_percentage:.2f}% of {total_hours} total hours)")

    return simulation_df


def filter_floods_by_region(flood_events: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Filter flood events by specific region within Kyrgyzstan.

    Args:
        flood_events: DataFrame from load_kyrgyzstan_floods()
        region: Region code ('chui', 'issyk_kul', 'osh', 'jalal_abad', etc.)

    Returns:
        Filtered DataFrame with events for the specified region
    """
    # Region keyword mapping
    region_keywords = {
        'chui': ['Chui', 'Chu', 'Bishkek', 'Bishkek City'],
        'issyk_kul': ['Issyk-Kul', 'Issyk Kul', 'Karakol', 'Issyk-Köl'],
        'naryn': ['Naryn'],
        'osh': ['Osh', 'Osh City'],
        'jalal_abad': ['Jalal-Abad', 'Jalal Abad', 'Jalalabad', 'Dzhalal-Abad'],
        'talas': ['Talas'],
        'batken': ['Batken']
    }

    keywords = region_keywords.get(region.lower(), [])

    if not keywords:
        logger.warning(f"Unknown region '{region}', returning all events")
        return flood_events

    # Filter by keywords in Country column
    mask = flood_events['Country'].str.contains('|'.join(keywords), case=False, na=False)
    filtered = flood_events[mask].copy()

    logger.info(f"Filtered to {len(filtered)} events for region '{region}'")

    return filtered


def get_flood_statistics(flood_events: pd.DataFrame) -> dict:
    """
    Calculate statistics for flood events.

    Args:
        flood_events: DataFrame from load_kyrgyzstan_floods()

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_events': len(flood_events),
        'date_range': {
            'start': flood_events['start_date'].min(),
            'end': flood_events['start_date'].max()
        },
        'severity_distribution': flood_events['severity'].value_counts().to_dict(),
        'total_deaths': flood_events['Dead'].sum(),
        'total_displaced': flood_events['Displaced'].sum(),
        'causes': flood_events['MainCause'].value_counts().head(5).to_dict()
    }

    return stats


# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Load flood events
    try:
        floods = load_kyrgyzstan_floods('data/FloodArchive.csv')

        print("\n" + "="*60)
        print("KYRGYZSTAN FLOOD DATABASE")
        print("="*60)

        print(f"\nTotal events: {len(floods)}")
        print(f"\nFirst few events:")
        print(floods.head(10).to_string())

        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)

        stats = get_flood_statistics(floods)
        print(f"\nTotal events: {stats['total_events']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"\nSeverity distribution:")
        for severity, count in stats['severity_distribution'].items():
            print(f"  {severity}: {count}")
        print(f"\nTotal deaths: {stats['total_deaths']}")
        print(f"Total displaced: {stats['total_displaced']}")
        print(f"\nMain causes:")
        for cause, count in stats['causes'].items():
            print(f"  {cause}: {count}")

    except FileNotFoundError as e:
        print(f"\n{e}")
