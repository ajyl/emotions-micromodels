"""
Utility functions to build empathy analysis component.
"""
from dash import html, dcc


def build_empathy_analysis_component(micromodel_results):
    """
    micromodel_results: {
        "epitome_er": {"max_score": float},
        ...
        "empathy_emotional_reactions": {"max_score": float},
        ...
    }
    """
    empathy_er = (
        round(
            micromodel_results["empathy_emotional_reactions"]["max_score"], 3
        ),
    )
    empathy_int = (
        round(micromodel_results["empathy_interpretations"]["max_score"], 3),
    )
    empathy_exp = (
        round(micromodel_results["empathy_explorations"]["max_score"], 3),
    )

    epitome_er = (round(micromodel_results["epitome_er"]["max_score"], 3),)
    epitome_int = (round(micromodel_results["epitome_int"]["max_score"], 3),)
    epitome_exp = (round(micromodel_results["epitome_exp"]["max_score"], 3),)
    return [
        html.Thead(
            html.Tr(
                [
                    html.Th(""),
                    html.Th("Emotional Reaction"),
                    html.Th("Interpretation"),
                    html.Th("Exploration"),
                ],
            ),
            style={"hidden": True},
        ),
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(
                            "Micromodels",
                            style={"font-weight": "bold"},
                        ),
                        html.Td(empathy_er),
                        html.Td(empathy_int),
                        html.Td(empathy_exp),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            "Epitome",
                            style={"font-weight": "bold"},
                        ),
                        html.Td(epitome_er),
                        html.Td(epitome_int),
                        html.Td(epitome_exp),
                    ]
                ),
            ]
        ),
    ]
