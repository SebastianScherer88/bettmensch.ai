import { Navigate, Route, Routes } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { PipelinesList } from "./pages/PipelinesList";
import { PipelineDetail } from "./pages/PipelineDetail";
import { RunsList } from "./pages/RunsList";
import { RunDetail } from "./pages/RunDetail";
import { ArtifactsSearch } from "./pages/ArtifactsSearch";

export default function App() {
  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      <Sidebar />
      <main className="flex-1 overflow-auto p-8">
        <Routes>
          <Route path="/" element={<Navigate to="/pipelines" replace />} />
          <Route path="/pipelines" element={<PipelinesList />} />
          <Route path="/pipelines/:pipelineName" element={<PipelineDetail />} />
          <Route path="/runs" element={<RunsList />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/artifacts" element={<ArtifactsSearch />} />
        </Routes>
      </main>
    </div>
  );
}
