from pathlib import Path

import pandas as pd
import plotly.express as px


# Source workbook and output folder
SOURCE_FILE = Path(
    r"C:\Users\user\OneDrive\Document\APU\FYP"
    r"\project_schedule_chapters_1_6.xlsx"
)
OUTPUT_DIR = Path(
    r"D:\APU\Micron\code -Micron(latest version)\outputs\gantt_chart"
)


def build_gantt(source_file: Path = SOURCE_FILE, output_dir: Path = OUTPUT_DIR):
    """Read the project schedule and save an interactive Gantt chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(source_file)
    df.columns = df.columns.str.strip()

    # Accept either the workbook's current headings or common alternatives.
    aliases = {
        "Name of Tasks": "Name of Task",
        "Task Name": "Name of Task",
        "Duration (Days)": "Duration",
    }
    df = df.rename(columns=aliases)

    required = {"Task ID", "Name of Task", "Start Date", "Duration"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["Start Date"] = pd.to_datetime(df["Start Date"], dayfirst=True, errors="raise")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="raise").astype(int)

    # Schedule durations are inclusive: a 1-day task starts and ends on the same day.
    calculated_end = df["Start Date"] + pd.to_timedelta(df["Duration"] - 1, unit="D")
    if "End Date" not in df.columns:
        df["End Date"] = calculated_end
    else:
        df["End Date"] = pd.to_datetime(df["End Date"], dayfirst=True, errors="coerce")
        df["End Date"] = df["End Date"].fillna(calculated_end)

    if "Status" not in df.columns:
        df["Status"] = "Not Specified"
    df["Status"] = df["Status"].fillna("Not Specified").astype(str).str.strip()

    # Plotly treats x_end as an exclusive boundary. Add one day visually so
    # same-day tasks have a visible bar while hover data shows the true end date.
    df["Visual End"] = df["End Date"] + pd.Timedelta(days=1)
    df["Task"] = df["Task ID"].astype(str) + "  " + df["Name of Task"].astype(str)
    df["Start"] = df["Start Date"].dt.strftime("%d %b %Y")
    df["End"] = df["End Date"].dt.strftime("%d %b %Y")

    status_order = ["Completed", "In Progress", "Not Started", "Not Specified"]
    colors = {
        "Completed": "#2E8B57",
        "In Progress": "#F4A261",
        "Not Started": "#6C83B5",
        "Not Specified": "#8A8A8A",
    }

    fig = px.timeline(
        df,
        x_start="Start Date",
        x_end="Visual End",
        y="Task",
        color="Status",
        # Preserve the workbook chapter/task sequence instead of grouping by status.
        category_orders={
            "Status": status_order,
            "Task": df["Task"].tolist()[::-1],
        },
        color_discrete_map=colors,
        custom_data=["Task ID", "Name of Task", "Start", "End", "Duration", "Status"],
        title="Project Gantt Chart",
        labels={"Task": "Project task"},
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]} 鈥?%{customdata[1]}</b><br>"
            "Start: %{customdata[2]}<br>"
            "End: %{customdata[3]}<br>"
            "Duration: %{customdata[4]} days<br>"
            "Status: %{customdata[5]}<extra></extra>"
        ),
        marker_line_color="rgba(25, 40, 65, 0.35)",
        marker_line_width=0.6,
    )
    fig.update_yaxes(
        autorange="reversed",
        title=None,
        tickfont={"size": 10},
        automargin=True,
    )
    fig.update_xaxes(
        title="Date",
        tickformat="%d %b\n%Y",
        dtick=7 * 24 * 60 * 60 * 1000,
        showgrid=True,
        gridcolor="rgba(120, 130, 150, 0.18)",
    )
    fig.update_layout(
        template="plotly_white",
        height=max(1050, 25 * len(df)),
        margin={"l": 390, "r": 45, "t": 90, "b": 60},
        legend={"title": "Status", "orientation": "h", "y": 1.035, "x": 0},
        title={"x": 0.01, "font": {"size": 22}},
        bargap=0.22,
        font={"family": "Arial, sans-serif"},
    )

    html_path = output_dir / "project_schedule_chapters_1_6_gantt.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

    # PNG export requires Plotly's Kaleido image engine. The HTML is always saved.
    png_path = output_dir / "project_schedule_chapters_1_6_gantt.png"
    try:
        fig.write_image(png_path, width=2200, height=max(1050, 25 * len(df)), scale=1)
    except Exception as exc:
        print(f"PNG export skipped: {exc}")
        png_path = None

    print(f"Saved interactive chart: {html_path}")
    if png_path:
        print(f"Saved PNG chart: {png_path}")
    return fig


if __name__ == "__main__":
    build_gantt()



