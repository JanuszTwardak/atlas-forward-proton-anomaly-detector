import uproot
from typing import List, Optional, Union
import pandas as pd
import gc
import os
from pathlib2 import Path
import logging
from warnings import simplefilter
import sys

log = logging.getLogger(__name__)


class RootHandler:
    """Class responsible for all root manipulation, especially data conversion from root to
    DataFrame type, including conversion from multi-column, fixed root data representation
    to modern DataFrame one"""

    @staticmethod
    def extract_root(
        root_paths: List[str],
        branches_to_extract: List[str],
        chunk_size: str,
        min_hits_no: int,
        max_hits_no: int,
        events_limit_no: Union[int, None],
        interim_dir: str,
    ) -> None:
        """extract_root Method that extracts root to DataFrame format, applies all functions that
        convert multi-column, fixed data representation into modern DataFrame style.
        Calculates standard deviation of hits' coordinates (spread), merges both station sides
        data, merges hits coming from 4 planes from single detector into average one (weighted
        average by charge).
        Extracted and processed data will be saved in '{interim_dir}/extracted_root.parquet' file.

        Args:
            root_paths (List[str]): List containing paths to root files that will be extracted.
            branches_to_extract (List[str]): List of branches that will be extracted from root
            TTree
            chunk_size (str): Size of single root chunk that will be processed at the time.
            min_hits_no (int): Minimal number of hits in single event. Any events with lower
            number of hits will be removed.
            max_hits_no (int): Maximum number of hits in single event. Any events with higher
            number of hits will be removed.
            events_limit_no (Union[int, None]): Limit number of events that will be extracted.
            If None, all events in file will be taken.
            interim_dir (str): Path to interim directory, where new folder "extracted_root"
            with function output will be created.
        """
        output_dir = Path(interim_dir) / "extracted_root"
        output_dir.mkdir(parents=True, exist_ok=True)

        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        sys.path.append(".")

        for root_path in root_paths:
            with uproot.open(root_path) as file:
                total_size = float(os.path.getsize(root_path)) * 1e-9
                chunk_iter = 0

                tree = file["TreeHits"]

                for chunk in tree.iterate(
                    branches_to_extract,
                    library="pd",
                    step_size=chunk_size,
                    entry_stop=events_limit_no,
                ):
                    chunk = RootHandler._extract_hits_number(chunk)
                    chunk = RootHandler._extract_average_coordinates(chunk)
                    chunk = RootHandler._extract_hit_std_deviation(chunk)
                    chunk = RootHandler._merge_sides(
                        chunk, min_hits_no, max_hits_no
                    )
                    chunk = RootHandler._merge_hit_std_deviations(chunk)

                    output_path = (
                        output_dir
                        / f"{Path(root_path).stem}_{chunk_iter}.parquet"
                    )
                    chunk.to_parquet(output_path, engine="pyarrow")

                    size_done = int(chunk_size[:-3]) * chunk_iter * 1e-3
                    log.info(
                        f"- preprocessing: {Path(root_path).stem} | progress: {size_done:.2f}/{total_size:.2f} GB"
                    )
                    log.debug(chunk.info(), chunk.memory_usage(deep=True))

                    chunk_iter += 1

        log.info("Preprocess finished!")

    @staticmethod
    def _extract_hits_number(df: pd.DataFrame) -> pd.DataFrame:
        """extract_hits_number Reduces redundant, multiple columns containing information about
        hit numbers into two single columns, each for stations side.

        Args:
            df (pd.DataFrame): Dataframe converted from root containing all columns with
            information about hit numbers.

        Returns:
            pd.DataFrame: Dataframe without redundant columns. They are replaced with two columns:
            'a_hits_n' and 'c_hits_n' containing total number of hits in single event for
            anti-clockwise and clockwise station sides, respectively.
        """

        df["a_hits_n"] = df.filter(regex="^hits\\[[01]", axis=1).sum(axis=1)
        df["c_hits_n"] = df.filter(regex="^hits\\[[23]", axis=1).sum(axis=1)
        df.drop(df.filter(regex="^hits\\[", axis=1), axis=1, inplace=True)
        return df

    @staticmethod
    def _extract_average_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """extract_average_coordinates Reduces multiple columns containing information about
        hit column and hit row of multiple hits in single event into two columns containing
        information about weighted average column and average row for all hits. Average is
        calculated with charge being weight for every hit.

        Args:
            df (pd.DataFrame): Dataframe converted from root containing all columns with
            information about hit columns and hit rows.

        Returns:
            pd.DataFrame: Dataframe without redundant columns. They are replaced with
            {X}_hit_row_1, {X}_hit_row_2, {X}_hit_column_1, {X}_hit_column_2,
            both for {X} = a and {X} = c (anticlockwise and clockwise station sides). Each
            column contains information about average value for 4 planes of adequate detector.
            Important: number 1 or 2 symbolizes order in which particle went through two detectors
            from one side, not order of station placement.
        """

        # first detector (in hit order, which means for side A we take detector #2 -> detector #1 data)
        weights_a_1 = df.filter(regex="^hits_q\\[1", axis=1).where(
            df.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
        )
        weights_c_1 = df.filter(regex="^hits_q\\[2", axis=1).where(
            df.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
        )
        rows_a = df.filter(regex="^hits_row\\[1", axis=1)
        rows_c = df.filter(regex="^hits_row\\[2", axis=1)
        df["a_hit_row_1"] = (rows_a * weights_a_1.values).sum(
            axis=1
        ) / weights_a_1.sum(axis=1)
        df["c_hit_row_1"] = (rows_c * weights_c_1.values).sum(
            axis=1
        ) / weights_c_1.sum(axis=1)

        # second detector (in hit order)
        weights_a_2 = df.filter(regex="^hits_q\\[0", axis=1).where(
            df.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
        )
        weights_c_2 = df.filter(regex="^hits_q\\[3", axis=1).where(
            df.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
        )
        df.drop(df.filter(regex="^hits_q", axis=1), axis=1, inplace=True)
        rows_a = df.filter(regex="^hits_row\\[0", axis=1)
        rows_c = df.filter(regex="^hits_row\\[3", axis=1)

        df["a_hit_row_2"] = (rows_a * weights_a_2.values).sum(
            axis=1
        ) / weights_a_2.sum(axis=1)

        df["c_hit_row_2"] = (rows_c * weights_c_2.values).sum(
            axis=1
        ) / weights_c_2.sum(axis=1)

        del [rows_a, rows_c]

        df[["a_hit_row_2", "c_hit_row_2"]] = df[
            ["a_hit_row_2", "c_hit_row_2"]
        ].apply(pd.to_numeric, downcast="unsigned")

        gc.collect()

        columns_a = df.filter(regex="^hits_col\\[1", axis=1)
        columns_c = df.filter(regex="^hits_col\\[2", axis=1)
        df["a_hit_column_1"] = (columns_a * weights_a_1.values).sum(
            axis=1
        ) / weights_a_1.sum(axis=1)
        df["c_hit_column_1"] = (columns_c * weights_c_1.values).sum(
            axis=1
        ) / weights_c_1.sum(axis=1)
        columns_a = df.filter(regex="^hits_col\\[0", axis=1)
        columns_c = df.filter(regex="^hits_col\\[3", axis=1)
        df["a_hit_column_2"] = (columns_a * weights_a_2.values).sum(
            axis=1
        ) / weights_a_2.sum(axis=1)
        df["c_hit_column_2"] = (columns_c * weights_c_2.values).sum(
            axis=1
        ) / weights_c_2.sum(axis=1)

        del [columns_a, columns_c]
        del [weights_a_1, weights_c_1, weights_a_2, weights_c_2]
        gc.collect()

        return df

    @staticmethod
    def _extract_hit_std_deviation(df: pd.DataFrame) -> pd.DataFrame:
        """extract_hit_std_deviation Calculated and adds columns to dataframe containing i
        nformation about standard deviation of hit column, row for each station side.

        Args:
            df (pd.DataFrame): Dataframe converted from root containing all columns with
            information about hit columns and hit rows.


        Returns:
            pd.DataFrame: Dataframe with extra columns: "{X}_std_col" and "{X}_std_row" for each
            {X} = a and {X} = c (anticlockwise and clockwise station sides) that contains
            respective standard deviation value of given variable.
        """
        df["a_std_col"] = df.filter(regex="^hits_col\\[[01]", axis=1).std(
            axis=1
        )
        df["a_std_row"] = df.filter(regex="^hits_row\\[[01]", axis=1).std(
            axis=1
        )

        df["c_std_col"] = df.filter(regex="^hits_col\\[[23]", axis=1).std(
            axis=1
        )
        df["c_std_row"] = df.filter(regex="^hits_row\\[[23]", axis=1).std(
            axis=1
        )

        df.drop(df.filter(regex="^hits_col", axis=1), axis=1, inplace=True)
        df.drop(df.filter(regex="^hits_row", axis=1), axis=1, inplace=True)

        return df

    @staticmethod
    def _merge_sides(
        df: pd.DataFrame,
        min_hits: Optional[int] = 1,
        max_hits: Optional[int] = 100,
    ) -> pd.DataFrame:
        """merge_detector_sides Combines columns containing same information about event
        that are separated for each of two station sides. This makes events indistinguishable
        in terms of side.

        Args:
            df (pd.DataFrame): Dataframe converted from root, after first preprocessing. it should
            have columns: "{X}_hits_n", "{X}_hit_row_1", "{X}_hit_row_2", "{X}_hit_column_1",
            "{X}_hit_column_2", "{X}_std_col", "{X}_std_row" for both {X} = a and {X} = c
            (anticlockwise and clockwise station sides).
            min_hits (Optional[int], optional): Minimum number of hits, otherwise event will
            be deleted from the dataframe. Defaults to 1.
            max_hits (Optional[int], optional): Maximum number of hits, otherwise event will
            be deleted from the dataframe. Defaults to 100.

        Returns:
            pd.DataFrame: Dataframe with merged columns from two station sides. Merged column
            names: "hits_n", "hit_row_1",, "hit_row_2", "hit_column_1", "hit_column_2",
            "_std_col", "_std_row".
        """

        buffor = df.drop(df.filter(regex="^c", axis=1), axis=1)
        buffor.rename(
            columns={
                "a_hits_n": "hits_n",
                "a_hit_row_1": "hit_row_1",
                "a_hit_row_2": "hit_row_2",
                "a_hit_column_1": "hit_column_1",
                "a_hit_column_2": "hit_column_2",
                "a_std_col": "_std_col",
                "a_std_row": "_std_row",
            },
            inplace=True,
        )
        buffor["side"] = "a"

        df = df.drop(df.filter(regex="^a", axis=1), axis=1)
        df.rename(
            columns={
                "c_hits_n": "hits_n",
                "c_hit_row_1": "hit_row_1",
                "c_hit_row_2": "hit_row_2",
                "c_hit_column_1": "hit_column_1",
                "c_hit_column_2": "hit_column_2",
                "c_std_col": "_std_col",
                "c_std_row": "_std_row",
            },
            inplace=True,
        )
        df["side"] = "c"
        df = df.append(buffor)
        df["side"] = df["side"].astype("category")

        df = df[df["hits_n"] >= min_hits]
        df = df[df["hits_n"] <= max_hits]

        return df

    @staticmethod
    def _merge_hit_std_deviations(df: pd.DataFrame) -> pd.DataFrame:
        """merge_std_deviations Replaces standard deviation for column and row with standard
        deviation distance, calculated with simple Pythagoras formula.

        Args:
            df (pd.DataFrame): Dataframe containing columns "_std_col" and "_std_row" containing
            information about standard deviation of hit columns and hit rows, respectively.

        Returns:
            pd.DataFrame: Dataframe with new column "std_distance".
        """

        df["std_distance"] = (
            df["_std_col"] * df["_std_col"].values
            + df["_std_row"].values * df["_std_row"].values
        )
        df["std_distance"] = df["std_distance"].pow(1 / 2)

        df.drop(df.filter(regex="^_", axis=1), axis=1, inplace=True)

        return df
