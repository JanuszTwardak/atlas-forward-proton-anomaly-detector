import uproot
from typing import List, Optional
import pandas as pd
import gc


class RootHandler:
    """Class responsible for all root manipulation, especially
    data conversion from root to DataFrame type."""

    def root_to_dataframe(
        root_paths: List[str],
    ):
        """root_to_dataframe Method converting dataset from root to DataFrame.


        Args:
            root_paths (_type_): _description_
        """
        preprocess_functions = []

        for single_root_name in root_files:
            root_path = str(
                os.path.join(
                    parameters.path_to_root_dict, single_root_name + ".root"
                )
            )

            with uproot.open(root_path) as file:
                total_size = float(os.path.getsize(root_path)) * 1e-9
                tree = file["TreeHits"]

                chunk_iter = 0

                for chunk in tree.iterate(
                    preprocess_branches, library="pd", step_size=chunk_size
                ):
                    chunk_iter += 1
                    file_name = single_root_name + str(chunk_iter)

                    chunk = Preprocessing.preprocess_single_dataframe(
                        chunk, preprocess_functions, single_root_name
                    )

                    Preprocessing.save_preprocessed_dataframe(
                        chunk, file_name=file_name
                    )

                    size_done = int(chunk_size[:-3]) * chunk_iter * 1e-3

                    print(
                        f"preprocessing: {single_root_name} | progress: {size_done:.2f}/{total_size:.2f} GB"
                    )
                    print(chunk)
                    print("@@ MEMORY USAGE @@", chunk.memory_usage(deep=True))
                    print(chunk.info())
        print("Preprocess finished!")

    def extract_hits_number(df: pd.DataFrame) -> pd.DataFrame:
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

    def extract_average_coordinates(df: pd.DataFrame) -> pd.DataFrame:
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

    def extract_hit_std_deviation(df: pd.DataFrame) -> pd.DataFrame:
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

    def merge_sides(
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

    def merge_hit_std_deviations(df: pd.DataFrame) -> pd.DataFrame:
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
