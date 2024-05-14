from pathlib import Path

import dash
from dash import dash_table, Input, Output, State, html, dcc

import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

from settings import AFTER_INCLUDE_ONLY, BINS, ORANGES
from options import OPTIONS, METRICS_VISUAL_RANGE

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Deep Value Strategy"
server = app.server
app.config["suppress_callback_exceptions"] = True

historical = (
    pd.read_csv(
        Path(__file__).parent / "data" / "historical_prices_monthly_stat.csv",
    )
    .dropna()
    .reset_index(drop=True)
)

meta = pd.read_csv(
    Path(__file__).parent / "data" / "meta.csv",
    dtype=str,
)

country_options = OPTIONS["country"] + [{"label": "Country - All", "value": "0"}]
sector_options = [o for o in OPTIONS["gics"] if len(o["value"]) == 2] + [
    {"label": "Sector - All", "value": "0"}
]


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Deep Value"),
                    html.H6("Analysis and Modeling for Deep Value Strategy"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Button(id="learn-more-button", children="About", n_clicks=0),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Overview-tab",
                        label="Overview",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        disabled=True,
                        disabled_style={"color": "#808080"},
                    ),
                    dcc.Tab(
                        id="EDA-tab",
                        label="EDA",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Result-tab",
                        label="Result",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        disabled=True,
                        disabled_style={"color": "#808080"},
                    ),
                ],
            )
        ],
    )


def build_overview():
    return [
        # Manually select metrics
        html.Div(
            id="set-overview-container",
            children=html.P("Overview (In Progress)"),
        ),
        html.Div(
            id="settings-menu",
            children=[
                html.Div(
                    id="metric-select-menu",
                    children=[
                        html.Label(id="metric-select-title", children="Select Metrics"),
                        html.Br(),
                    ],
                ),
                html.Div(
                    id="metric-select-menu",
                    children=[
                        html.Label(id="metric-select-title", children="Select Metrics"),
                        html.Br(),
                    ],
                ),
            ],
        ),
    ]


def generate_modal():
    return html.Div(
        id="markdown",
        className="modal",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                        ###### ...

                      ...

                    """
                            )
                        ),
                    ),
                ],
            )
        ),
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        children=[
            generate_section_banner("Filter"),
            html.Div(
                id="card-0",
                children=[
                    html.P("Period"),
                    dcc.RangeSlider(
                        id="quick-stats-period-rangeslider",
                        min=2014,
                        max=2024,
                        step=1,
                        value=[2014, 2024],
                        marks={
                            2014: "2014",
                            2017: "2017",
                            2020: "2020",
                            2022: "2022",
                            2024: "2024",
                        },
                    ),
                ],
            ),
            html.Div(
                id="card-1",
                children=[
                    html.P("Country"),
                    dcc.Dropdown(
                        id="quick-stats-country-dropdown",
                        options=country_options,
                        value="0",
                        searchable=True,
                    ),
                    html.P("GICS"),
                    dcc.Dropdown(
                        id="quick-stats-sector-dropdown",
                        options=sector_options,
                        value="0",
                        searchable=True,
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        id="quick-stats-industry-group-dropdown",
                        searchable=True,
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        id="quick-stats-industry-dropdown",
                        searchable=True,
                    ),
                ],
            ),
            # generate_section_banner("Select Number of Bins"),
            html.Div(
                id="card-2",
                children=[
                    html.P("Number of Bins"),
                    dcc.RadioItems(
                        id="quick-stats-bins-raioitems",
                        options=OPTIONS["bins"],
                        value="10",
                        inline=True,
                    ),
                ],
            ),
            # generate_section_banner("Select Metrics & Statistics"),
            html.Div(
                id="card-3",
                children=[
                    html.P("Metrics"),
                    dcc.Dropdown(
                        id="quick-stats-metrics-dropdown",
                        options=OPTIONS["metrics"],
                        value=["monthly_start_high_rtn"],
                        multi=True,
                        clearable=False,
                    ),
                    html.P("Statistics"),
                    dcc.Dropdown(
                        id="quick-stats-statistics-dropdown",
                        options=OPTIONS["statistics"],
                        value=["mean"],
                        multi=True,
                        clearable=False,
                    ),
                ],
            ),
        ],
    )


@app.callback(
    Output("quick-stats-industry-group-dropdown", "options"),
    Output("quick-stats-industry-group-dropdown", "value"),
    Input("quick-stats-sector-dropdown", "value"),
)
def render_quick_stats_industry_group_dropdown(value):
    if value == "0":
        ops = []
    else:
        ops = [
            o
            for o in OPTIONS["gics"]
            if (len(o["value"]) == 4) & (o["value"].startswith(value))
        ]
    ops.append({"label": "Industry Group - All", "value": "0"})
    return ops, ops[-1]["value"]


@app.callback(
    Output("quick-stats-industry-dropdown", "options"),
    Output("quick-stats-industry-dropdown", "value"),
    Input("quick-stats-industry-group-dropdown", "value"),
)
def render_quick_stats_industry_dropdown(value):
    if value == "0":
        ops = []
    else:
        ops = [
            o
            for o in OPTIONS["gics"]
            if (len(o["value"]) == 6) & (o["value"].startswith(value))
        ]
    ops.append({"label": "Industry - All", "value": "0"})
    return ops, ops[-1]["value"]


def build_top_panel():
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # 8width graph
            html.Div(
                id="metric-summary-session",
                className="nine columns",
                children=[
                    generate_section_banner("Bar Chart"),
                    dcc.Graph(id="eda-bar-chart"),
                ],
            ),
            html.Div(
                id="count-summary-session",
                className="three columns",
                children=[
                    generate_section_banner("Count"),
                    html.Table(id="count-table"),
                ],
            ),
        ],
    )


def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="row",
        children=[
            html.Div(
                id="distplot-session",
                className="six columns",
                children=[
                    generate_section_banner("KDE Plot"),
                    dcc.Graph(id="eda-distplot"),
                ],
            ),
            html.Div(
                id="not-yet",
                className="six columns",
                children=[
                    generate_section_banner("-"),
                ],
            ),
        ],
    )


@app.callback(
    Output(component_id="eda-bar-chart", component_property="figure"),
    Output(component_id="count-table", component_property="children"),
    Output(component_id="eda-distplot", component_property="figure"),
    Input(component_id="quick-stats-period-rangeslider", component_property="value"),
    Input(component_id="quick-stats-country-dropdown", component_property="value"),
    Input(component_id="quick-stats-sector-dropdown", component_property="value"),
    Input(
        component_id="quick-stats-industry-group-dropdown", component_property="value"
    ),
    Input(component_id="quick-stats-industry-dropdown", component_property="value"),
    Input(component_id="quick-stats-bins-raioitems", component_property="value"),
    [Input(component_id="quick-stats-metrics-dropdown", component_property="value")],
    [Input(component_id="quick-stats-statistics-dropdown", component_property="value")],
)
def rendor_eda_bar_chart(vp, vc, vs, vig, vi, vb, vm, vst):

    # Filtering

    if vm == []:
        vm = ["monthly_start_high_rtn"]
    if vst == []:
        vst = ["mean"]

    filtered_meta = meta
    filtered_meta = (
        filtered_meta if vc == "0" else filtered_meta[filtered_meta["country"] == vc]
    )
    filtered_meta = (
        filtered_meta
        if vs == "0"
        else filtered_meta[filtered_meta["gics_sector"] == vs]
    )
    filtered_meta = (
        filtered_meta
        if vig == "0"
        else filtered_meta[filtered_meta["gics_industry_group"] == vig]
    )
    filtered_meta = (
        filtered_meta
        if vi == "0"
        else filtered_meta[filtered_meta["gics_industry"] == vi]
    )
    df = pd.merge(
        historical[(historical["_year"] >= vp[0]) & (historical["_year"] <= vp[1])],
        filtered_meta,
        how="inner",
        on="_code",
    )

    if AFTER_INCLUDE_ONLY:
        df = df[
            pd.to_datetime(
                df["_year"].astype(str) + df["_month"].astype(str).str.rjust(2, "0"),
                format="%Y%m",
            )
            >= df["first_include"]
        ]

    bs = BINS[vb]
    colors = [ORANGES[int(cl)] for cl in np.linspace(0, len(ORANGES) - 1, len(bs))]
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]

    df = df.sort_values(["_code", "_year", "_month"], ascending=True).reset_index(
        drop=True
    )

    df["monthly_high_end_rtn_category"] = pd.cut(
        df["monthly_high_end_rtn"], bins=bs, labels=lbls
    ).astype(str)
    df["before_monthly_high_end_rtn"] = df.groupby("_code", as_index=False)[
        "monthly_high_end_rtn"
    ].shift(1)
    df["before_monthly_high_end_rtn_category"] = df.groupby("_code", as_index=False)[
        "monthly_high_end_rtn_category"
    ].shift(1)

    # Make Bar Chart

    df_result = df.groupby("before_monthly_high_end_rtn_category")[vm].agg(vst)

    m_titles = {d["value"]: d["label"] for d in OPTIONS["metrics"]}
    st_titles = {d["value"]: d["label"] for d in OPTIONS["statistics"]}

    fig = make_subplots(
        rows=len(vst),
        cols=len(vm),
        start_cell="top-left",
        subplot_titles=[f"{m_titles[m]} - {st_titles[st]}" for st in vst for m in vm],
    )

    for midx, m in enumerate(vm):
        for stidx, st in enumerate(vst):
            fig.add_trace(
                go.Bar(
                    y=df_result[(m, st)],
                    x=df_result[(m, st)].index,
                    marker={
                        "color": colors,
                    },
                ),
                row=stidx + 1,
                col=midx + 1,
            )

    fig.for_each_xaxis(
        lambda x: x.update(showline=False, showgrid=False, zeroline=False)
    )
    fig.for_each_yaxis(
        lambda x: x.update(showline=False, showgrid=False, zeroline=False)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        margin=dict(t=50, r=10, b=30, l=10),
        showlegend=False,
        font=dict(color="rgba(255,255,255,255)"),
    )

    # Make Table

    df_count = (
        df.groupby("before_monthly_high_end_rtn_category")["monthly_start_high_rtn"]
        .count()
        .reset_index(drop=False)
    )

    df_count["pct"] = (
        "("
        + (df_count.iloc[:, -1] / sum(df_count.iloc[:, -1]))
        .multiply(100)
        .round(2)
        .astype(str)
        + "%)"
    )

    table = []
    for _, row in df_count.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row.iloc[i]]))
        table.append(html.Tr(html_row))

    # Make KDE Plot

    fig2 = ff.create_distplot(
        [
            df[df["before_monthly_high_end_rtn_category"] == l][vm[0]]
            for l in reversed(lbls)
        ],
        [l for l in reversed(lbls)],
        show_hist=False,
        colors=colors,
        show_rug=False,
        histnorm="probability",
    )

    fig2.update_xaxes(showgrid=False, range=METRICS_VISUAL_RANGE[vm[0]])
    fig2.update_yaxes(
        showgrid=False,
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        margin=dict(t=50, r=10, b=30, l=10),
        showlegend=True,
        font=dict(color="rgba(255,255,255,255)"),
    )

    return fig, table, fig2


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                html.Div(id="app-content"),
            ],
        ),
        generate_modal(),
    ],
)


@app.callback(
    Output("app-content", "children"),
    Input("app-tabs", "value"),
)
def render_tab_content(tab_switch):
    if tab_switch == "tab1":
        return build_overview()
    elif tab_switch == "tab2":
        return html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_top_panel(), build_chart_panel()],
                ),
            ],
        )


# ======= Callbacks for modal popup =======
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "learn-more-button":
            return {"display": "block"}

    return {"display": "none"}


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
