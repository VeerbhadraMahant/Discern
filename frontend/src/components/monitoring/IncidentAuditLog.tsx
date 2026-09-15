import { useState } from "react";
import { SAMPLE_AUDIT_LOGS } from "../../lib/sampleScenarios";
import type { AuditRecord } from "../../lib/types";
import { FileText, Search, Download, Wand2 } from "lucide-react";

interface Props {
  onSelectAuditPreset?: (presetId: string) => void;
}

export default function IncidentAuditLog({ onSelectAuditPreset }: Props) {
  const [logs, setLogs] = useState<AuditRecord[]>(SAMPLE_AUDIT_LOGS);
  const [searchQuery, setSearchQuery] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("all");

  const filteredLogs = logs.filter((log) => {
    const matchesSearch =
      log.cameraName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.violationType.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.id.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesSeverity =
      severityFilter === "all" || log.severity === severityFilter;

    return matchesSearch && matchesSeverity;
  });

  const toggleStatus = (id: string) => {
    setLogs((prev) =>
      prev.map((item) => {
        if (item.id === id) {
          const nextStatus: AuditRecord["status"] =
            item.status === "open"
              ? "acknowledged"
              : item.status === "acknowledged"
              ? "resolved"
              : "open";
          return { ...item, status: nextStatus };
        }
        return item;
      })
    );
  };

  return (
    <div className="space-y-6 font-mono text-hermes-ink">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-hermes-ink/15">
        <div>
          <div className="flex items-center gap-2 text-xs">
            <FileText className="h-4 w-4 text-hermes-blue" />
            <span className="uppercase tracking-widest text-hermes-muted font-bold">
              // REGULATORY_LOGS
            </span>
          </div>
          <h1 className="mt-1 hermes-title text-2xl sm:text-3xl font-bold uppercase text-hermes-ink">
            Site Incident Ledger
          </h1>
        </div>

        {/* Export Button */}
        <button
          onClick={() => {
            const dataStr = JSON.stringify(logs, null, 2);
            const blob = new Blob([dataStr], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `discern-audit-ledger-${Date.now()}.json`;
            a.click();
          }}
          className="hermes-btn-primary bg-hermes-blue text-white text-xs flex items-center gap-2"
        >
          <Download className="h-3.5 w-3.5" />
          <span>Export Ledger (.JSON)</span>
        </button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-3 text-xs">
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-hermes-muted" />
          <input
            type="text"
            placeholder="FILTER BY CAMERA, HAZARD TYPE, ZONE, ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white pl-10 pr-4 py-2.5 text-xs text-hermes-ink placeholder:text-hermes-muted border border-hermes-ink/20 focus:outline-none focus:border-hermes-blue uppercase"
          />
        </div>

        {/* Severity Tabs */}
        <div className="flex items-center gap-1 border border-hermes-ink/20 p-1 bg-white uppercase">
          <button
            onClick={() => setSeverityFilter("all")}
            className={`px-3 py-1.5 font-bold transition-all ${
              severityFilter === "all" ? "bg-hermes-blue text-white" : "text-hermes-charcoal"
            }`}
          >
            :ALL_{logs.length}:
          </button>
          <button
            onClick={() => setSeverityFilter("critical")}
            className={`px-3 py-1.5 font-bold transition-all ${
              severityFilter === "critical" ? "bg-hermes-critical text-white" : "text-hermes-critical"
            }`}
          >
            :CRITICAL:
          </button>
          <button
            onClick={() => setSeverityFilter("warning")}
            className={`px-3 py-1.5 font-bold transition-all ${
              severityFilter === "warning" ? "bg-hermes-warning text-white" : "text-hermes-warning"
            }`}
          >
            :WARNING:
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden border border-hermes-ink/20 bg-white shadow-lift">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-hermes-ink/20 bg-hermes-paper text-[11px] font-bold uppercase tracking-wider text-hermes-muted">
                <th className="py-3 px-4">RECORD_ID</th>
                <th className="py-3 px-4">TIMESTAMP</th>
                <th className="py-3 px-4">CAMERA / ZONE</th>
                <th className="py-3 px-4">WEATHER</th>
                <th className="py-3 px-4">RESTORATION</th>
                <th className="py-3 px-4">HAZARD</th>
                <th className="py-3 px-4 text-center">SEVERITY</th>
                <th className="py-3 px-4 text-right">STATUS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hermes-ink/10">
              {filteredLogs.map((log) => (
                <tr key={log.id} className="hover:bg-hermes-paper/60 transition-colors">
                  <td className="py-3 px-4 font-bold text-hermes-blue">
                    {log.id}
                  </td>
                  <td className="py-3 px-4 text-hermes-muted">
                    {log.timestamp}
                  </td>
                  <td className="py-3 px-4">
                    <span className="font-bold text-hermes-ink block">{log.cameraName}</span>
                    <span className="text-[11px] text-hermes-muted">{log.zone}</span>
                  </td>
                  <td className="py-3 px-4 text-hermes-charcoal">
                    <span className="font-bold text-hermes-ink">{log.weather}</span>
                    <span className="text-[11px] text-hermes-muted block">{log.lighting}</span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="bg-hermes-paper border border-hermes-ink/15 px-2 py-0.5 text-[11px] font-bold block truncate max-w-[150px]">
                      {log.restorationApplied}
                    </span>
                  </td>
                  <td className="py-3 px-4 max-w-xs">
                    <span className="font-bold text-hermes-ink block">{log.violationType}</span>
                    <span className="text-[11px] text-hermes-charcoal leading-tight block mt-0.5 font-body line-clamp-2">
                      {log.description}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span
                      className={`px-2 py-0.5 text-[11px] font-bold uppercase ${
                        log.severity === "critical"
                          ? "bg-hermes-critical text-white"
                          : log.severity === "warning"
                          ? "bg-hermes-warning text-white"
                          : "bg-hermes-ink text-white"
                      }`}
                    >
                      {log.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      onClick={() => toggleStatus(log.id)}
                      className={`px-2.5 py-1 text-[11px] font-bold uppercase cursor-pointer transition-all ${
                        log.status === "open"
                          ? "bg-hermes-critical text-white"
                          : log.status === "acknowledged"
                          ? "bg-hermes-warning text-white"
                          : "bg-hermes-safe text-white"
                      }`}
                    >
                      :{log.status.toUpperCase()}:
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
