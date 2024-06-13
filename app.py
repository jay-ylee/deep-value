from pathlib import Path

import dash
from dash import dash_table, Input, Output, State, html, dcc

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

from settings import AFTER_INCLUDE_ONLY, ORANGES
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


def generate_little_banner(title):
    return html.Div(className="little-banner", children=title)


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
                        clearable=False,
                        searchable=True,
                        optionHeight=40,
                    ),
                    html.P("GICS"),
                    dcc.Dropdown(
                        id="quick-stats-sector-dropdown",
                        options=sector_options,
                        value="0",
                        clearable=False,
                        searchable=True,
                        optionHeight=40,
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        id="quick-stats-industry-group-dropdown",
                        clearable=False,
                        searchable=True,
                        optionHeight=40,
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        id="quick-stats-industry-dropdown",
                        clearable=False,
                        searchable=True,
                        optionHeight=40,
                    ),
                ],
            ),
            generate_section_banner("View"),
            html.Div(
                id="card-2",
                children=[
                    html.P("Return View Type"),
                    dcc.RadioItems(
                        id="quick-stats-return-type",
                        value="exp",
                        options=OPTIONS["viewtype"],
                    ),
                    html.P("Bins (Input Type: Arithmetic)"),
                    dcc.Input(
                        id="quick-stats-bins",
                        value="-0.4, -0.3, -0.2, -0.1, -0.05",
                        debounce=True,
                        placeholder="ex. -0.4, -0.1, -0.01",
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


def is_float(s: str):
    try:
        s = float(s)
        return True
    except:
        return False


def bins_text_to_list(txt: str, bound: str | None = 'upper'):
    bns = [float(t) for t in txt.replace(' ', '').split(',') if is_float(t)]
    if bns == []:
        return [-np.inf, np.inf]
    elif bound == 'upper':
        bns = [v for v in bns if v <=0.] + [-np.inf, 0.]
        bns = np.sort(np.unique(bns))
    elif bound == 'lower':
        bns = [v for v in bns if v >=0.] + [0., np.inf]
        bns = np.sort(np.unique(bns))
    else:
        bns = np.sort(np.unique(np.insert(bns, [0, len(bns)], [-np.inf, np.inf])))
    return bns.tolist()


@app.callback(
    Output(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-period-rangeslider", component_property="value"),
    Input(component_id="quick-stats-country-dropdown", component_property="value"),
    Input(component_id="quick-stats-sector-dropdown", component_property="value"),
    Input(
        component_id="quick-stats-industry-group-dropdown", component_property="value"
    ),
    Input(component_id="quick-stats-industry-dropdown", component_property="value"),
)
def filter_meta_data(prid, ctry, stor, itrygrp, itry):

    filtered_meta = meta.copy()
    filtered_meta = (
        filtered_meta
        if ctry == "0"
        else filtered_meta[filtered_meta["country"] == ctry]
    )
    filtered_meta = (
        filtered_meta
        if stor == "0"
        else filtered_meta[filtered_meta["gics_sector"] == stor]
    )
    filtered_meta = (
        filtered_meta
        if itrygrp == "0"
        else filtered_meta[filtered_meta["gics_industry_group"] == itrygrp]
    )
    filtered_meta = (
        filtered_meta
        if itry == "0"
        else filtered_meta[filtered_meta["gics_industry"] == itry]
    )

    df = pd.merge(
        historical[(historical["_year"] >= prid[0]) & (historical["_year"] <= prid[1])],
        filtered_meta[["_code", "first_include"]],
        how="inner",
        on="_code",
    )
    if AFTER_INCLUDE_ONLY:
        df = df[
            pd.to_datetime(
                df["_year"].astype(str) + df["_month"].astype(str).str.rjust(2, "0"),
                format="%Y%m",
            )
            >= pd.to_datetime(df["first_include"], format="%Y-%m-%d")
        ]
    df = df.sort_values(["_code", "_year", "_month"], ascending=True).reset_index(
        drop=True
    )

    return df.to_dict("records")


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
                    generate_section_banner("Simple Analysis - Bar Chart"),
                    html.Div(
                        id="eda-bar-chart-dropdowns",
                        children=[
                            generate_little_banner("Statistic | "),
                            dcc.Dropdown(
                                id = "eda-bar-chart-dropdown1",
                                options=OPTIONS["statistics"],
                                value="mean",
                                clearable=False,
                                searchable=True,
                                placeholder="Select Group Statistic",
                                optionHeight=40,
                            ),
                            generate_little_banner("| Metric (Sub) | "),
                            dcc.Dropdown(
                                id = "eda-bar-chart-dropdown2",
                                options=OPTIONS["metrics"],
                                value="1mf_monthly_rtn",
                                clearable=False,
                                searchable=True,
                                placeholder="Select SubMetric",
                                optionHeight=60,
                            ),
                        ]
                    ),
                    dcc.Loading(
                        children=[dcc.Graph(id="eda-bar-chart")],
                    ),
                ],
            ),
            html.Div(
                id="count-summary-session",
                className="three columns",
                children=[
                    generate_section_banner("Count"),
                    dcc.Loading(
                        children=[html.Table(id="count-table")],
                    ),
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
                id="tsplot-session",
                className="six columns",
                children=[
                    generate_section_banner("Monthly Historical Analysis - Line Chart"),
                    html.Div(
                        id="eda-ts-chart-dropdowns",
                        children=[
                            generate_little_banner("Metric | "),
                            dcc.Dropdown(
                                id="eda-ts-chart-dropdown1",
                                options=OPTIONS["metrics"],
                                value="1mf_monthly_start_high_rtn",
                                clearable=False,
                                searchable=True,
                                placeholder="Select Metric",
                                optionHeight=60,
                            ),
                        ]
                    ),
                    dcc.Loading(
                        children=[dcc.Graph(id="eda-tsplot")],
                    ),
                ],
            ),
            html.Div(
                id="heatmap-session",
                className="six columns",
                children=[
                    generate_section_banner("Metrics CrossTab - Heatmap"),
                    html.Div(
                        id="eda-heatmap-dropdowns",
                        children=[
                            dcc.Dropdown(
                                id="eda-heatmap-dropdown1",
                                options=OPTIONS["metrics"],
                                value="monthly_start_high_rtn",
                                clearable=False,
                                searchable=True,
                                placeholder="Select SubMetric",
                                optionHeight=60,
                            ),
                            dcc.Input(
                                id="eda-heatmap-text1",
                                value="0.05, 0.1, 0.2, 0.3, 0.4",
                                debounce=True,
                                placeholder="ex. 0.01, 0.1, 0.4",
                            ),
                            generate_little_banner(" Y "),
                            generate_little_banner("|"),
                            generate_little_banner(" Z "),
                            dcc.Dropdown(
                                id="eda-heatmap-dropdown2",
                                options=OPTIONS["metrics"],
                                value="1mf_monthly_start_high_rtn",
                                clearable=False,
                                searchable=True,
                                placeholder="Select SubMetric",
                                optionHeight=60,
                            ),
                        ]
                    ),
                    dcc.Loading(
                        children=[dcc.Graph(id="eda-heatmap")],
                    ),
                ],
            ),
        ],
    )


def build_dist_panel():
    return html.Div(
        id="dist-section-container",
        className="row",
        children=[
            # 8width graph
            html.Div(
                id="dist-session",
                className="nine columns",
                children=[
                    generate_section_banner("Distribution Plot"),
                    html.Div(
                        id="eda-dist-dropdowns",
                        children=[
                            generate_little_banner("Metric | "),
                            dcc.Dropdown(
                                id = "eda-dist-dropdown1",
                                options=OPTIONS["metrics"],
                                value="1mf_monthly_rtn",
                                clearable=False,
                                searchable=True,
                                placeholder="Select Metric",
                                optionHeight=60,
                            ),
                        ]
                    ),
                    # dcc.Loading(
                    #     children=[dcc.Graph(id="eda-bar-chart")],
                    # ),
                ],
            ),
            # html.Div(
            #     id="count-summary-session",
            #     className="three columns",
            #     children=[
            #         generate_section_banner("Count"),
            #         dcc.Loading(
            #             children=[html.Table(id="count-table")],
            #         ),
            #     ],
            # ),
        ],
    )


@app.callback(
    Output(component_id="eda-bar-chart", component_property="figure"),
    Input(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-return-type", component_property="value"),
    Input(component_id="quick-stats-bins", component_property="value"),
    Input(component_id="eda-bar-chart-dropdown2", component_property="value"),
    Input(
        component_id="eda-bar-chart-dropdown1", component_property="value"
    ),
)
def rendor_eda_bar_chart(hdata, tpe:str, bns: str, mtrc: str, stt: str):

    mtrcs = ["1mf_monthly_start_high_rtn", mtrc]
    use_cols = [
        "_code",
        "_year",
        "_month",
        "monthly_high_end_rtn",
        "monthly_start_high_rtn",
    ]
    if mtrc.replace("1mf_", "") not in ["monthly_high_end_rtn", "monthly_start_high_rtn"]:
        use_cols += [mtrc.replace("1mf_", "")]

    df = pd.DataFrame.from_records(hdata)[use_cols]

    df = (
        pd.concat(
            [
                df,
                df.groupby("_code", as_index=False)
                .shift(-1)
                .rename(columns={c: "1mf_" + c for c in df.columns}),
            ],
            axis=1,
        )
        .dropna()
        .reset_index(drop=True)
    )
    bs = bins_text_to_list(bns)
    colors = [ORANGES[int(cl)] for cl in np.linspace(0, len(ORANGES) - 1, len(bs))]
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]

    df["monthly_high_end_rtn_category"] = pd.cut(
        np.exp(df["monthly_high_end_rtn"])-1., bins=bs, labels=lbls
    ).astype(str)

    # Make Bar Chart

    if tpe == 'exp':
        df[mtrcs] = np.exp(df[mtrcs]) - 1.
    df = df.groupby("monthly_high_end_rtn_category")[mtrcs].agg(stt)

    m_titles = {d["value"]: d["label"] for d in OPTIONS["metrics"]}
    st_titles = {d["value"]: d["label"] for d in OPTIONS["statistics"]}

    fig = make_subplots(
        rows=1,
        cols=2,
        start_cell="top-left",
        subplot_titles=[f"{m_titles[m]} - {st_titles[stt]}" for m in mtrcs],
    )

    for midx, m in enumerate(mtrcs):
        fig.add_trace(
            go.Bar(
                y=df[m],
                x=df[m].index,
                marker={
                    "color": colors,
                },
                name=m_titles[m],
            ),
            row=1,
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
        margin=dict(t=30, r=10, b=30, l=10),
        showlegend=False,
        font=dict(color="rgba(255,255,255,255)"),
    )

    return fig


@app.callback(
    Output(component_id="count-table", component_property="children"),
    Input(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-bins", component_property="value"),
)
def rendor_count_table(hdata, bns):

    df = pd.DataFrame.from_records(hdata)[
        ["_code", "_year", "_month", "monthly_high_end_rtn"]
    ]

    df = (
        pd.concat(
            [
                df,
                df.groupby("_code", as_index=False)
                .shift(-1)
                .rename(columns={c: "1mf_" + c for c in df.columns}),
            ],
            axis=1,
        )
        .dropna()
        .reset_index(drop=True)
    )

    bs = bins_text_to_list(bns)
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]

    df["monthly_high_end_rtn_category"] = pd.cut(
        np.exp(df["monthly_high_end_rtn"])-1., bins=bs, labels=lbls
    ).astype(str)

    df_count = (
        df.groupby("monthly_high_end_rtn_category")["monthly_high_end_rtn"]
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

    return table


@app.callback(
    Output(component_id="eda-tsplot", component_property="figure"),
    Input(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-return-type", component_property="value"),
    Input(component_id="quick-stats-bins", component_property="value"),
    Input(component_id="eda-ts-chart-dropdown1", component_property="value"),
)
def rendor_ts_chart(hdata, tpe:str, bns: str, mtrc: str):

    use_cols = [
        "_code",
        "_year",
        "_month",
        "monthly_high_end_rtn",
    ]

    if mtrc.replace("1mf_", "") != "monthly_high_end_rtn":
        use_cols += [mtrc.replace("1mf_", "")]

    df = pd.DataFrame.from_records(hdata)[use_cols]

    df = pd.concat([df, 
                    df.groupby("_code", as_index=False).shift(-1).rename(columns={c: "1mf_" + c for c in df.columns})],
                    axis=1,).dropna().reset_index(drop=True)

    bs = bins_text_to_list(bns)
    colors = [ORANGES[int(cl)] for cl in np.linspace(0, len(ORANGES) - 1, len(bs)+1)]
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]

    df["monthly_high_end_rtn_category"] = pd.cut(
        np.exp(df["monthly_high_end_rtn"])-1., bins=bs, labels=lbls
    ).astype(str)

    df['ym'] = pd.to_datetime(df["_year"].astype(str) + df["_month"].astype(str).str.rjust(2, "0"), format="%Y%m")

    # Make TS Chart

    if tpe == 'exp':
        df[mtrc] = np.exp(df[mtrc]) - 1.

    m_titles = {d["value"]: d["label"] for d in OPTIONS["metrics"]}

    total_grp = df.groupby(['ym'], as_index=False
                           )[mtrc].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    total_grp['monthly_high_end_rtn_category'] = 'All'
    bins_grp = df.groupby(['monthly_high_end_rtn_category', 'ym'], as_index=False
                          )[mtrc].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    grp = pd.concat([total_grp, bins_grp]).reset_index(drop=True).fillna(0)

    fig = px.line(grp, x="ym", y="mean", color="monthly_high_end_rtn_category",
              markers=True,
              color_discrete_sequence=colors,
              hover_name="monthly_high_end_rtn_category",
              hover_data={'monthly_high_end_rtn_category': False,
                          'ym': False,
                          'count': True,
                          'mean': ':.4f',
                          'median': ':.4f',
                          'std': ':.4f'},)
    fig.update_traces(marker={'size': 5})
    fig.update_xaxes(title="Time", showline=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title=f"{m_titles[mtrc]} - Mean", 
                     showline=False, 
                     showgrid=False, 
                     zeroline=False,)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        margin=dict(t=30, r=10, b=30, l=10),
        showlegend=False,
        font=dict(color="rgba(255,255,255,255)"),
    )

    return fig


app.layout = html.Div(
    id="big-app-container",
    children=[
        dcc.Store(id="historical-data"),
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
    Output(component_id="eda-heatmap", component_property="figure"),
    Input(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-return-type", component_property="value"),
    Input(component_id="quick-stats-bins", component_property="value"),
    Input(component_id="eda-heatmap-dropdown1", component_property="value"),
    Input(component_id="eda-heatmap-text1", component_property="value"),
    Input(component_id="eda-heatmap-dropdown2", component_property="value"),
)
def rendor_heatmap(hdata, tpe:str, bns: str, yv: str, ybns: str, zv: str):

    use_cols = [
        "_code",
        "_year",
        "_month",
        "monthly_high_end_rtn",
    ]

    if yv.replace("1mf_", "") != "monthly_high_end_rtn":
        use_cols += [yv.replace("1mf_", "")]
    if zv.replace("1mf_", "") not in ["monthly_high_end_rtn", yv.replace("1mf_", "")]:
        use_cols += [zv.replace("1mf_", "")]
    df = pd.DataFrame.from_records(hdata)[use_cols]

    df = pd.concat([df, 
                    df.groupby("_code", as_index=False).shift(-1).rename(columns={c: "1mf_" + c for c in df.columns})],
                    axis=1,).dropna().reset_index(drop=True)

    bs = bins_text_to_list(bns)
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]
    df["monthly_high_end_rtn_category"] = pd.cut(
        np.exp(df["monthly_high_end_rtn"])-1., bins=bs, labels=lbls
    ).astype(str)

    bound = None
    if (df[yv] >= 0.).all():
        bound = "lower"
    elif (df[yv] <= 0.).all():
        bound = "upper"
    ybs = bins_text_to_list(ybns, bound)

    if bound == "lower":
        rgt = False
        ylbls = [f"[{ybs[i-1]}, {ybs[i]})" for i, _ in enumerate(ybs) if i > 0]
    else:
        rgt = True
        ylbls = [f"({ybs[i-1]}, {ybs[i]}]" for i, _ in enumerate(ybs) if i > 0]
    df[f"{yv}_category"] = pd.cut(
        np.exp(df[yv])-1., bins=ybs, labels=ylbls, right=rgt,
    ).astype(str)

    df['ym'] = pd.to_datetime(df["_year"].astype(str) + df["_month"].astype(str).str.rjust(2, "0"), format="%Y%m")

    if tpe == 'exp':
        df[zv] = np.exp(df[zv]) - 1.

    m_titles = {d["value"]: d["label"] for d in OPTIONS["metrics"]}

    ctmean = pd.crosstab(
        df[f"{yv}_category"], 
        df['monthly_high_end_rtn_category'], 
        df[zv], 
        aggfunc='mean'
    )

    ctcount = pd.crosstab(
        df[f"{yv}_category"], 
        df['monthly_high_end_rtn_category'], 
        df[zv], 
        aggfunc='count'
    ).to_numpy()

    ctmedian = pd.crosstab(
        df[f"{yv}_category"], 
        df['monthly_high_end_rtn_category'], 
        df[zv], 
        aggfunc='median'
    ).to_numpy()

    ctstd = pd.crosstab(
        df[f"{yv}_category"], 
        df['monthly_high_end_rtn_category'], 
        df[zv], 
        aggfunc='std'
    ).fillna(0.).to_numpy()

    fig = go.Figure(
        data=go.Heatmap(
            name="mean",
            z=ctmean.values,
            x=ctmean.columns,
            y=ctmean.index,
            colorscale='Blues',
            customdata=np.dstack((ctcount, ctmedian, ctstd)),
            hovertemplate='%{z:.4f}<br>count: %{customdata[0]:.4f}<br>median: %{customdata[1]:.4f}<br>std: %{customdata[2]:.4f}',
            hoverongaps = False,
        ))
    
    fig.update_xaxes(title=f"{m_titles['monthly_high_end_rtn']}")
    fig.update_yaxes(title=f"{m_titles[yv]}")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        margin=dict(t=30, r=5, b=30, l=10),
        showlegend=False,
        font=dict(color="rgba(255,255,255,255)"),
    )

    return fig


@app.callback(
    Output(component_id="eda-dist-plot", component_property="figure"),
    Input(component_id="historical-data", component_property="data"),
    Input(component_id="quick-stats-return-type", component_property="value"),
    Input(component_id="quick-stats-bins", component_property="value"),
    Input(component_id="eda-bar-chart-dropdown2", component_property="value"),
    Input(
        component_id="eda-bar-chart-dropdown1", component_property="value"
    ),
)
def rendor_dist_plot(hdata, tpe:str, bns: str, mtrc: str, stt: str):

    mtrcs = ["1mf_monthly_start_high_rtn", mtrc]
    use_cols = [
        "_code",
        "_year",
        "_month",
        "monthly_high_end_rtn",
        "monthly_start_high_rtn",
    ]
    if mtrc.replace("1mf_", "") not in ["monthly_high_end_rtn", "monthly_start_high_rtn"]:
        use_cols += [mtrc.replace("1mf_", "")]

    df = pd.DataFrame.from_records(hdata)[use_cols]

    df = (
        pd.concat(
            [
                df,
                df.groupby("_code", as_index=False)
                .shift(-1)
                .rename(columns={c: "1mf_" + c for c in df.columns}),
            ],
            axis=1,
        )
        .dropna()
        .reset_index(drop=True)
    )
    bs = bins_text_to_list(bns)
    colors = [ORANGES[int(cl)] for cl in np.linspace(0, len(ORANGES) - 1, len(bs))]
    lbls = [f"({bs[i-1]}, {bs[i]}]" for i, _ in enumerate(bs) if i > 0]

    df["monthly_high_end_rtn_category"] = pd.cut(
        np.exp(df["monthly_high_end_rtn"])-1., bins=bs, labels=lbls
    ).astype(str)

    # Make Bar Chart

    if tpe == 'exp':
        df[mtrcs] = np.exp(df[mtrcs]) - 1.
    df = df.groupby("monthly_high_end_rtn_category")[mtrcs].agg(stt)

    m_titles = {d["value"]: d["label"] for d in OPTIONS["metrics"]}
    st_titles = {d["value"]: d["label"] for d in OPTIONS["statistics"]}

    fig = make_subplots(
        rows=1,
        cols=2,
        start_cell="top-left",
        subplot_titles=[f"{m_titles[m]} - {st_titles[stt]}" for m in mtrcs],
    )

    for midx, m in enumerate(mtrcs):
        fig.add_trace(
            go.Bar(
                y=df[m],
                x=df[m].index,
                marker={
                    "color": colors,
                },
                name=m_titles[m],
            ),
            row=1,
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
        margin=dict(t=30, r=10, b=30, l=10),
        showlegend=False,
        font=dict(color="rgba(255,255,255,255)"),
    )

    return fig


app.layout = html.Div(
    id="big-app-container",
    children=[
        dcc.Store(id="historical-data"),
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
                    children=[build_top_panel(), build_chart_panel(), build_dist_panel()],
                ),
            ],
        )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
