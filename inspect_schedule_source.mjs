import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "outputs/gantt_chart/project_schedule_chapters_1_6.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const overview = await workbook.inspect({
  kind: "workbook,sheet,table,drawing",
  maxChars: 12000,
  tableMaxRows: 20,
  tableMaxCols: 10,
});
console.log(overview.ndjson);
