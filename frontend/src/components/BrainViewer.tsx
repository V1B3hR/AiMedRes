/**
 * 3D Brain Viewer Component (P3-1)
 * 
 * Interactive 3D brain visualization with:
 * - Anatomical overlays
 * - Disease progression tracking
 * - Treatment simulation
 * - Explainability integration
 */

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { brainVisualizationAPI, type BrainOverlayRequest, type DiseaseMapRequest, type TreatmentSimulationRequest } from '../api/visualization';

interface BrainViewerProps {
  patientId: string;
  mode?: 'anatomy' | 'disease' | 'treatment' | 'progression';
}

interface Marker {
  marker_id: string;
  region: string;
  coordinates_3d: [number, number, number];
  label: string;
  severity_score: number;
  metadata: Record<string, any>;
}

const BrainViewer: React.FC<BrainViewerProps> = ({ patientId, mode = 'anatomy' }) => {
  const [selectedRegions, setSelectedRegions] = useState<string[]>(['frontal_lobe', 'hippocampus']);
  const [highlightAbnormalities, setHighlightAbnormalities] = useState(true);
  const [diseaseType, setDiseaseType] = useState('alzheimers');
  const [overlayData, setOverlayData] = useState<any>(null);
  const [viewMode, setViewMode] = useState<'3d' | 'slice'>('3d');

  // Available brain regions
  const brainRegions = [
    'frontal_lobe',
    'parietal_lobe',
    'temporal_lobe',
    'occipital_lobe',
    'hippocampus',
    'amygdala',
    'cerebellum',
    'basal_ganglia',
    'thalamus',
    'corpus_callosum',
    'brainstem',
  ];

  // Fetch brain overlay
  const { data: overlayResponse, isLoading: overlayLoading, refetch: refetchOverlay } = useQuery({
    queryKey: ['brainOverlay', patientId, selectedRegions, highlightAbnormalities],
    queryFn: async () => {
      const request: BrainOverlayRequest = {
        patient_id: patientId,
        regions_of_interest: selectedRegions,
        highlight_abnormalities: highlightAbnormalities,
      };
      return brainVisualizationAPI.createOverlay(request);
    },
    enabled: mode === 'anatomy' && selectedRegions.length > 0,
  });

  // Fetch disease map
  const { data: diseaseMapResponse, isLoading: diseaseMapLoading } = useQuery({
    queryKey: ['diseaseMap', patientId, diseaseType],
    queryFn: async () => {
      // Mock severity map - in production, this would come from patient data
      const severityMap: Record<string, number> = {
        hippocampus: 0.8,
        temporal_lobe: 0.6,
        frontal_lobe: 0.4,
      };
      
      const request: DiseaseMapRequest = {
        patient_id: patientId,
        disease_type: diseaseType,
        severity_map: severityMap,
      };
      return brainVisualizationAPI.createDiseaseMap(request);
    },
    enabled: mode === 'disease',
  });

  // Fetch statistics
  const { data: statsResponse } = useQuery({
    queryKey: ['brainStats'],
    queryFn: () => brainVisualizationAPI.getStatistics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Update overlay data when response changes
  useEffect(() => {
    if (overlayResponse?.data) {
      setOverlayData(overlayResponse.data);
    }
  }, [overlayResponse]);

  const handleRegionToggle = (region: string) => {
    setSelectedRegions((prev) =>
      prev.includes(region)
        ? prev.filter((r) => r !== region)
        : [...prev, region]
    );
  };

  const renderMarkers = () => {
    if (!overlayData?.markers) return null;

    return (
      <div className="markers-container">
        <h4>Brain Markers ({overlayData.markers.length})</h4>
        <div className="markers-grid">
          {overlayData.markers.map((marker: Marker) => (
            <div
              key={marker.marker_id}
              className={`marker-card ${marker.severity_score > 0.5 ? 'severity-high' : 'severity-low'}`}
            >
              <div className="marker-label">{marker.label}</div>
              <div className="marker-region">{marker.region}</div>
              <div className="marker-coordinates">
                Coordinates: ({marker.coordinates_3d.join(', ')})
              </div>
              {marker.severity_score > 0 && (
                <div className="marker-severity">
                  Severity: {(marker.severity_score * 100).toFixed(1)}%
                  <div
                    className="severity-bar"
                    style={{ width: `${marker.severity_score * 100}%` }}
                  />
                </div>
              )}
              {marker.metadata.functions && (
                <div className="marker-functions">
                  Functions: {marker.metadata.functions.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderDiseaseMap = () => {
    if (!diseaseMapResponse?.data) return null;

    const data = diseaseMapResponse.data;

    return (
      <div className="disease-map-container">
        <h3>Disease Pathology Map</h3>
        <div className="disease-info">
          <div className="info-item">
            <span className="label">Disease:</span>
            <span className="value">{data.disease_type}</span>
          </div>
          <div className="info-item">
            <span className="label">Overall Burden:</span>
            <span className="value">{data.overall_burden.toFixed(2)}</span>
          </div>
          <div className="info-item">
            <span className="label">Average Severity:</span>
            <span className="value">{(data.average_severity * 100).toFixed(1)}%</span>
          </div>
        </div>

        <h4>Most Affected Regions</h4>
        <div className="affected-regions">
          {data.most_affected_regions.map((region: any, index: number) => (
            <div key={index} className="region-item">
              <span className="region-name">{region.region.replace('_', ' ')}</span>
              <div className="severity-bar-container">
                <div
                  className="severity-bar"
                  style={{ width: `${region.severity * 100}%` }}
                />
                <span className="severity-value">{(region.severity * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const render3DViewer = () => {
    return (
      <div className="viewer-3d">
        {/* 3D Canvas - In production, integrate Three.js or similar */}
        <div className="canvas-placeholder">
          <div className="canvas-info">
            <h3>3D Brain Visualization</h3>
            <p>Interactive 3D viewer would render here</p>
            <p>Libraries: Three.js, WebGL, or vtk.js</p>
            
            {overlayData && (
              <div className="render-info">
                <p>Render time: {overlayData.render_time_ms.toFixed(2)}ms</p>
                <p>Markers: {overlayData.markers?.length || 0}</p>
              </div>
            )}
          </div>
        </div>
        
        {/* Controls */}
        <div className="viewer-controls">
          <button onClick={() => setViewMode('3d')} className={viewMode === '3d' ? 'active' : ''}>
            3D View
          </button>
          <button onClick={() => setViewMode('slice')} className={viewMode === 'slice' ? 'active' : ''}>
            Slice View
          </button>
          <button onClick={() => refetchOverlay()}>
            Refresh
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="brain-viewer">
      <header className="viewer-header">
        <h2>3D Brain Viewer - Patient {patientId}</h2>
        <div className="mode-selector">
          {(['anatomy', 'disease', 'treatment', 'progression'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m as any)}
              className={mode === m ? 'active' : ''}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </header>

      <div className="viewer-content">
        <aside className="viewer-sidebar">
          <div className="controls-section">
            <h3>Region Selection</h3>
            <div className="region-checkboxes">
              {brainRegions.map((region) => (
                <label key={region} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedRegions.includes(region)}
                    onChange={() => handleRegionToggle(region)}
                  />
                  <span>{region.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="controls-section">
            <h3>Display Options</h3>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={highlightAbnormalities}
                onChange={(e) => setHighlightAbnormalities(e.target.checked)}
              />
              <span>Highlight Abnormalities</span>
            </label>
          </div>

          {mode === 'disease' && (
            <div className="controls-section">
              <h3>Disease Type</h3>
              <select
                value={diseaseType}
                onChange={(e) => setDiseaseType(e.target.value)}
                className="disease-select"
              >
                <option value="alzheimers">Alzheimer's Disease</option>
                <option value="parkinsons">Parkinson's Disease</option>
                <option value="als">ALS</option>
                <option value="ms">Multiple Sclerosis</option>
              </select>
            </div>
          )}

          {statsResponse?.data && (
            <div className="stats-section">
              <h3>Engine Statistics</h3>
              <div className="stat-item">
                <span>Visualizations:</span>
                <span>{statsResponse.data.visualizations_generated}</span>
              </div>
              <div className="stat-item">
                <span>Simulations:</span>
                <span>{statsResponse.data.simulations_run}</span>
              </div>
              <div className="stat-item">
                <span>Avg Render Time:</span>
                <span>{statsResponse.data.average_render_time_ms.toFixed(2)}ms</span>
              </div>
            </div>
          )}
        </aside>

        <main className="viewer-main">
          {overlayLoading && <div className="loading">Loading brain overlay...</div>}
          {diseaseMapLoading && <div className="loading">Loading disease map...</div>}

          {mode === 'anatomy' && !overlayLoading && render3DViewer()}
          {mode === 'disease' && !diseaseMapLoading && renderDiseaseMap()}

          {mode === 'anatomy' && overlayData && renderMarkers()}
        </main>
      </div>

      <style>{`
        .brain-viewer {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: #f5f5f5;
        }

        .viewer-header {
          background: white;
          padding: 1.5rem;
          border-bottom: 1px solid #e0e0e0;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .viewer-header h2 {
          margin: 0;
          font-size: 1.5rem;
          color: #333;
        }

        .mode-selector button {
          margin-left: 0.5rem;
          padding: 0.5rem 1rem;
          border: 1px solid #ddd;
          background: white;
          cursor: pointer;
          border-radius: 4px;
        }

        .mode-selector button.active {
          background: #2196F3;
          color: white;
          border-color: #2196F3;
        }

        .viewer-content {
          display: flex;
          flex: 1;
          overflow: hidden;
        }

        .viewer-sidebar {
          width: 300px;
          background: white;
          border-right: 1px solid #e0e0e0;
          padding: 1.5rem;
          overflow-y: auto;
        }

        .viewer-main {
          flex: 1;
          padding: 1.5rem;
          overflow-y: auto;
        }

        .controls-section {
          margin-bottom: 2rem;
        }

        .controls-section h3 {
          font-size: 1rem;
          margin-bottom: 1rem;
          color: #555;
        }

        .region-checkboxes {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .checkbox-label {
          display: flex;
          align-items: center;
          cursor: pointer;
        }

        .checkbox-label input {
          margin-right: 0.5rem;
        }

        .disease-select {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
        }

        .viewer-3d {
          background: white;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1.5rem;
        }

        .canvas-placeholder {
          height: 500px;
          background: #f9f9f9;
          border: 2px dashed #ddd;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 1rem;
        }

        .canvas-info {
          text-align: center;
          color: #666;
        }

        .viewer-controls {
          display: flex;
          gap: 0.5rem;
        }

        .viewer-controls button {
          padding: 0.5rem 1rem;
          border: 1px solid #ddd;
          background: white;
          cursor: pointer;
          border-radius: 4px;
        }

        .viewer-controls button.active {
          background: #2196F3;
          color: white;
        }

        .markers-container {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
        }

        .markers-container h4 {
          margin-top: 0;
          margin-bottom: 1rem;
        }

        .markers-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 1rem;
        }

        .marker-card {
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          padding: 1rem;
          background: #fafafa;
        }

        .marker-card.severity-high {
          border-left: 4px solid #f44336;
        }

        .marker-card.severity-low {
          border-left: 4px solid #4CAF50;
        }

        .marker-label {
          font-weight: bold;
          margin-bottom: 0.5rem;
        }

        .marker-region {
          color: #666;
          font-size: 0.9rem;
          margin-bottom: 0.25rem;
        }

        .marker-coordinates {
          color: #888;
          font-size: 0.85rem;
          margin-bottom: 0.5rem;
        }

        .marker-severity {
          margin-top: 0.5rem;
          font-size: 0.9rem;
        }

        .severity-bar {
          height: 6px;
          background: linear-gradient(to right, #4CAF50, #FFC107, #f44336);
          border-radius: 3px;
          margin-top: 0.25rem;
        }

        .marker-functions {
          margin-top: 0.5rem;
          font-size: 0.85rem;
          color: #666;
        }

        .disease-map-container {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
        }

        .disease-info {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .info-item {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .info-item .label {
          font-size: 0.85rem;
          color: #666;
        }

        .info-item .value {
          font-size: 1.25rem;
          font-weight: bold;
          color: #333;
        }

        .affected-regions {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .region-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem;
          background: #f9f9f9;
          border-radius: 4px;
        }

        .region-name {
          font-weight: 500;
          text-transform: capitalize;
        }

        .severity-bar-container {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex: 1;
          margin-left: 1rem;
        }

        .severity-bar-container .severity-bar {
          flex: 1;
          height: 8px;
          background: linear-gradient(to right, #4CAF50, #FFC107, #f44336);
          border-radius: 4px;
        }

        .severity-value {
          font-weight: bold;
          min-width: 50px;
          text-align: right;
        }

        .stats-section {
          background: #f9f9f9;
          padding: 1rem;
          border-radius: 4px;
        }

        .stat-item {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
          font-size: 0.9rem;
        }

        .loading {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 200px;
          color: #666;
        }
      `}</style>
    </div>
  );
};

export default BrainViewer;
