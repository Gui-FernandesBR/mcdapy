__author__ = "Guilherme Fernandes Alves"
__email__ = "gf10.alves@gmail.com"

import flet as ft
import matplotlib.pyplot as plt
import numpy as np
import json
from flet.matplotlib_chart import MatplotlibChart

from .methods.pattern import Pattern


class App:
    """Class to create the application

    Attributes
    ----------
    page: ft.Page
        The main page object

    Examples
    --------
    >>> import mcdapy as mcd
    >>> run = mcd.App("DESKTOP")

    """

    def __init__(self, theme="light") -> None:
        # Initialize the app

        self.theme = theme
        ft.app(target=self.main)

        return None

    def main(self, page: ft.Page) -> None:

        # Store the inputted information
        self.__initialize_main(page)

        # Modify basic properties of the page
        self.__set_layout_options()

        # Add tabs to the page
        self.__add_tabs()

        # Define references, which will be used to retrieve values
        self.__define_references()

        # Define input forms
        self.__define_forms()

        # Add contents to initialization page
        self.__add_init_contents()

        # Add app bar and float button
        self.__add_app_bar()
        self.__add_float_button()

        # Final update
        self.page.update()

        return None

    def __initialize_main(self, page):
        self.page = page

        self.init_container = ft.Container(
            content=ft.Column(
                controls=[], horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
        self.results_container = ft.Container(
            content=ft.Column(
                controls=[], horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
        self.help_container = ft.Container(
            content=ft.Column(
                controls=[], horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
        self.values = {}

        self.themes = {
            "light": ft.ThemeMode.LIGHT,
            "dark": ft.ThemeMode.DARK,
        }

        return None

    def __run_pattern(self) -> None:
        # Run the PATTERN method

        self.init_container.content.controls.append(
            ft.Text("Running PATTERN method. Checkout the results tab!", size=14)
        )

        self.page.update()

        pattern = Pattern(
            alternatives=self.alternatives_names,
            criteria=self.criteria_names,
            matrix=self.values["matrix"],
            weights=self.values["weights"],
        )
        pattern.solve()

        self.solution = pattern

        self.__add_results_contents()

        return None

    def __set_layout_options(self) -> None:
        # Set the basic settings for the page

        self.page.title = "MCDAPy v0.1.0 by Guilherme Fernandes Alves"
        self.page.scroll = "always"
        self.page.window_maximized = False
        self.page.window_width = 600
        self.page.window_height = 600
        self.page.window_top = 0
        self.page.window_center()
        self.page.theme_mode = self.themes.get(self.theme, ft.ThemeMode.LIGHT)

        return None

    def __define_references(self) -> None:
        # Define text fields references, it is used to retrieve the values
        self.references = {
            "n_rows": ft.Ref[ft.TextField](),
            "n_cols": ft.Ref[ft.TextField](),
            "method": ft.Ref[ft.Dropdown](),
            "matrix": ft.Ref[ft.ListView](),
            "weights": ft.Ref[ft.ListView](),
            "criteria_types": ft.Ref[ft.ListView](),
            "matrix_fig": ft.Ref[MatplotlibChart](),
            "rank_fig": ft.Ref[MatplotlibChart](),
        }
        return None

    def __check_inputs(self) -> None:
        # Retrieve values from the text fields, ensuring the inputs are valid

        try:
            self.values["n_rows"] = int(self.references["n_rows"].current.value)
        except ValueError:
            raise ValueError("Number of rows must be an integer")
        try:
            self.values["n_cols"] = int(self.references["n_cols"].current.value)
        except ValueError:
            raise ValueError("Number of columns must be an integer")

        return None

    def __create_matrix(self) -> None:
        # Create the Matrix when the Generate Matrix button is clicked

        self.__check_inputs()
        # Get some nicknames
        rows, cols = (self.values["n_rows"], self.values["n_cols"])

        self.alternatives_names = tuple(f"Alter. {i+1}" for i in range(rows))
        self.criteria_names = tuple(f"Crit. {i+1}" for i in range(cols))

        # If a matrix already exists, remove it
        self.init_container.content.controls = self.init_container.content.controls[:4]

        # Print a message to the user saying the matrix is being generated
        self.init_container.content.controls.insert(
            4,
            ft.Text(
                f"Setting up a matrix within {rows} rows and {cols} columns, a total of {rows * cols} cells!",
                size=14,
            ),
        )

        # Initialize the matrix
        lv = ft.ListView(controls=[], auto_scroll=True, ref=self.references["matrix"])

        # Add the header
        header = ft.Row(
            controls=[ft.Text("", width=100)],
            wrap=False,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        for i in range(cols):
            header.controls.append(
                ft.Text(f"Crit. {i+1}", width=80, weight="bold", text_align="center")
            )
        lv.controls.append(header)

        # Add the alternatives' rows
        for i in range(rows):
            row = ft.Row(
                [
                    ft.Text(
                        f"Alter. {i+1}", width=100, text_align="center", weight="bold"
                    )
                ],
                wrap=False,
                alignment=ft.MainAxisAlignment.CENTER,
            )
            for _ in range(cols):
                row.controls.append(
                    ft.TextField(label=f"", width=80, height=35, text_align="right")
                )
            lv.controls.append(row)

        # Add weights to the matrix
        row = ft.Row(
            controls=[
                ft.Text(
                    "Weights", width=100, text_align="center", weight="bold", size=11
                )
            ],
            wrap=False,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        for _ in range(cols):
            row.controls.append(
                ft.TextField(
                    label=f"",
                    width=80,
                    height=35,
                    text_align="right",
                )
            )
        c1 = ft.Container(content=row, padding=20)
        lv.controls.append(c1)

        self.init_container.content.controls.append(lv)

        # Add a container to hold the checkboxes

        # Add the checkboxes

        row = ft.Row(
            controls=[
                ft.Text(
                    "Benefit",
                    weight="bold",
                    width=100,
                    tooltip="Do higher values mean better? You need to answer this question for each criterion",
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            wrap=False,
        )
        for i in range(cols):
            self.references[f"checkbox_{i}"] = ft.Ref[ft.Checkbox]()
            row.controls.append(
                ft.Container(
                    ft.Checkbox(
                        ref=self.references[f"checkbox_{i}"],
                        value=True,
                    ),
                    width=80,
                    # alignment=ft.MainAxisAlignment.CENTER,
                )
            )

        self.init_container.content.controls.append(row)

        # Add the save matrix button
        self.init_container.content.controls.append(
            ft.ElevatedButton(
                "Save Matrix",
                on_click=lambda e: self.__save_matrix(),
                icon=ft.icons.SAVE,
                # icon_color=ft.colors.GREEN,
                elevation=15,
            )
        )

        self.page.update()

        return None

    def __save_matrix(self) -> None:
        # Retrieve values from the matrix and store in an array

        # Get some nicknames
        rows, cols = (self.values["n_rows"], self.values["n_cols"])

        # Initialize the matrix
        matrix = np.zeros((rows, cols))

        # Retrieve the values from the matrix
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = float(
                    self.references["matrix"]
                    .current.controls[i + 1]
                    .controls[j + 1]
                    .value
                )

        # Retrieve the weights from the matrix
        weights = np.zeros(cols)
        for j in range(cols):
            weights[j] = float(
                self.references["matrix"]
                .current.controls[rows + 1]
                .content.controls[j + 1]
                .value
            )

        # Print a message to the user saying the matrix is being generated
        # ref["greetings"].current.controls = []
        self.init_container.content.controls.append(
            ft.Text(
                f"Matrix saved!",
                size=14,
            )
        )

        # Invert the matrix if the criteria are not benefits
        for i in range(cols):
            if not self.references[f"checkbox_{i}"].current.value:
                matrix[:, i] = 1 / matrix[:, i]

        self.values["matrix"] = matrix
        self.values["weights"] = weights

        def run_trigger() -> None:
            # Select the method to be ran
            if self.references["method"].current.value == "ELECTRE":
                self.__run_pattern()  # TODO: Change this to the ELECTRE method
            else:
                self.__run_pattern()
            return None

        # Add a button to run the solution
        self.init_container.content.controls.append(
            ft.ElevatedButton(
                "Run",
                on_click=lambda e: run_trigger(),
                icon_color=ft.colors.GREEN,
                tooltip='Click to run the solution, the results will be shown in the "Results" tab',
                icon=ft.icons.START,
                elevation=15,
            )
        )
        self.page.update()

        return None

    def __define_forms(self) -> None:

        # Text fields to define the number of rows and columns
        n_rows = ft.TextField(
            ref=self.references["n_rows"],
            label="Number of alternatives",
            width=200,
            text_size=14,
        )
        n_cols = ft.TextField(
            ref=self.references["n_cols"],
            label="Number of criteria",
            width=200,
            text_size=14,
        )

        # Dropdown menu with to select the method to be used
        method = ft.Dropdown(
            label="Select the method",
            options=[ft.dropdown.Option("PATTERN"), ft.dropdown.Option("ELECTRE")],
            ref=self.references["method"],
            width=200,
            text_size=14,
        )

        # An elevated button to generate the matrix when clicked
        gen_matrix = ft.ElevatedButton(
            "Generate Matrix",
            on_click=lambda e: self.__create_matrix(),
            icon=ft.icons.DRAW,
            elevation=15,
        )

        # Store all forms in a dictionary
        self.forms = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "method": method,
            "gen_matrix": gen_matrix,
        }

        return None

    def __add_float_button(self) -> None:
        # Add the refresh button to the lower-right corner of the page
        def reset_pressed(e):
            self.results_container.content.controls = (
                self.results_container.content.controls[:3]
            )
            self.init_container.content.controls = self.init_container.content.controls[
                :4
            ]
            self.page.update()
            return None

        self.page.floating_action_button = ft.FloatingActionButton(
            icon=ft.icons.REFRESH,
            on_click=lambda e: reset_pressed(e),
            tooltip="Restart the application",
        )

        return None

    def __add_app_bar(self):
        def reverse_theme(e):
            if self.page.theme_mode == ft.ThemeMode.LIGHT:
                self.page.theme_mode = ft.ThemeMode.DARK
            else:
                self.page.theme_mode = ft.ThemeMode.LIGHT
            self.page.update()

        self.page.appbar = ft.AppBar(
            leading=ft.Icon(ft.icons.SCHOOL),
            leading_width=40,
            title=ft.Text(
                "MCDAPy - Multi-Criteria Decision Analysis in Python",
                size=14,
                weight="bold",
            ),
            center_title=False,
            elevation=15,
            actions=[
                ft.IconButton(
                    icon=ft.icons.WB_SUNNY_OUTLINED,
                    on_click=reverse_theme,
                    tooltip="Change theme mode",
                ),
                ft.IconButton(
                    icon=ft.icons.STAR,
                    on_click=lambda e: self.page.launch_url(
                        "https://github.com/Gui-FernandesBR/mcdapy"
                    ),
                    tooltip="Star the project on GitHub",
                ),
                ft.IconButton(
                    icon=ft.icons.QUESTION_MARK,
                    on_click=lambda e: self.page.launch_url(
                        "https://github.com/Gui-FernandesBR/mcdapy"
                    ),
                    tooltip="See the documentation",
                ),
                ft.IconButton(
                    icon=ft.icons.BUG_REPORT,
                    on_click=lambda e: self.page.launch_url(
                        "https://github.com/Gui-FernandesBR/mcdapy/issues/new"
                    ),
                    tooltip="Report a bug",
                ),
            ],
        )
        return None

    def __add_tabs(self) -> None:
        # Create the three tabs on the top of the page

        t = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Initialize",
                    icon=ft.icons.HOME,
                    content=self.init_container,
                ),
                ft.Tab(
                    text="Results",
                    icon=ft.icons.ANALYTICS,
                    content=self.results_container,
                ),
                # ft.Tab(
                #     text="Help",
                #     icon=ft.icons.HELP_CENTER,
                #     content=self.help_container,
                # ),
            ],
        )

        self.page.add(t)

        return None

    def __add_init_contents(self):
        # Add the forms to the initialization page
        self.init_container.content.controls.append(
            ft.Row(
                controls=[
                    self.forms["n_rows"],
                    self.forms["n_cols"],
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
        self.init_container.content.controls.append(self.forms["method"])
        self.init_container.content.controls.append(self.forms["gen_matrix"])

        # Print a title in the page
        self.init_container.content.controls.insert(
            0,
            ft.Row(
                controls=[
                    ft.Text(
                        value="Welcome to MCDAPy!",
                        size=25,
                        weight="bold",
                    ),
                ],
                alignment="center",
            ),
        )

        # Add headers to the results page
        self.results_container.content.controls.append(
            ft.Text(
                "Welcome to the results page!",
                size=25,
                weight="bold",
            )
        )
        self.results_container.content.controls.append(
            ft.Text(
                "The results of your analysis are shown below.",
                size=14,
            )
        )

        self.results_container.content.controls.append(
            ft.Row(
                controls=[
                    ft.FilledButton(
                        "Export results",
                        icon=ft.icons.DOWNLOAD,
                        on_click=lambda e: self.__download_results(),
                        tooltip="Export a report summarizing the results",
                        disabled=False,
                    )
                ],
                alignment="center",
            )
        )

        return None

    def __add_results_contents(self):

        self.results_container.content.controls.append(
            ft.Text(
                f"The best alternative is: {self.solution.ranking_named[0]}",
                size=18,
                weight="bold",
            )
        )

        self.results_container.content.controls.append(ft.Row([], height=20))

        self.results_container.content.controls.append(
            ft.Text(
                f"The worst alternative is: {self.solution.ranking_named[-1]}",
                size=18,
                # weight="bold",
            )
        )
        self.results_container.content.controls.append(ft.Row([], height=20))

        self.results_container.content.controls.append(
            ft.Text(
                f"The rank of alternatives is: {', '.join(map(str,self.solution.ranking_named))}",
                size=18,
                # weight="bold",
            )
        )
        self.results_container.content.controls.append(ft.Row([], height=20))

        self.results_container.content.controls.append(
            ft.Text(
                "See the weighted normalized matrix below:",
                size=14,
            )
        )

        fig, _ = self.solution.plot_matrix(return_fig=True)

        self.results_container.content.controls.append(
            MatplotlibChart(
                fig, expand=False, original_size=True, ref=self.references["matrix_fig"]
            )
        )
        plt.close("all")
        self.results_container.content.controls.append(ft.Row([], height=20))

        self.results_container.content.controls.append(
            ft.Text(
                "See the Pertinence Index Ranked below:",
                size=14,
            )
        )
        fig, _ = self.solution.plot_rank(return_fig=True)

        self.results_container.content.controls.append(
            MatplotlibChart(
                fig, expand=False, original_size=True, ref=self.references["rank_fig"]
            )
        )
        plt.close("all")

        self.page.update()

        return None

    def __download_results(self):
        # Download the results of the analysis

        # Return None if self.solution doesn't exist
        try:
            self.solution
        except AttributeError:
            return None

        # Create a dictionary with the results
        self.results = {
            "n_rows": self.values["n_rows"],
            "n_cols": self.values["n_cols"],
            "method": self.references["method"].current.value,
            "matrix": self.values["matrix"].tolist(),
            "normalized_matrix": self.solution.normalized_matrix.tolist(),
            "weights": self.values["weights"].tolist(),
            "normalized_weights": self.solution.normalized_weights.tolist(),
            "criteria": self.criteria_names,
            "alternatives": self.alternatives_names,
            "ranking": self.solution.ranking.tolist(),
            "ranking_named": self.solution.ranking_named,
        }

        # Save the dictionary as a JSON file
        with open("results.json", "w") as f:
            json.dump(self.results, f)

        # Save the plot as a PNG file
        self.references["matrix_fig"].current.figure.savefig("matrix_fig.png")

        # Show a message to the user
        self.results_container.content.controls.append(
            ft.Text(
                "The results were successfully exported, check the 'results.json' and 'matrix_fig.png' files in your working directory",
                size=14,
                color="green",
            )
        )
        self.page.update()

        return None
