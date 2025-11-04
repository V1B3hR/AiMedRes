/**
 * DICOM Viewer Component (P3-1)
 * 
 * Medical imaging viewer with:
 * - DICOM series browsing
 * - Multi-planar reconstruction
 * - Window/level adjustments
 * - Explainability overlays
 */

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { dicomViewerAPI } from '../api/visualization';

interface DicomViewerProps {
  patientId: string;
}

interface DicomSeries {
  series_id: string;
  patient_id: string;
  modality: string;
  series_description: string;
  study_date: string;
  num_instances: number;
  series_number: number;
}

const DicomViewer: React.FC<DicomViewerProps> = ({ patientId }) => {
  const [selectedSeries, setSelectedSeries] = useState<string | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [windowCenter, setWindowCenter] = useState<number | undefined>();
  const [windowWidth, setWindowWidth] = useState<number | undefined>();
  const [showExplainability, setShowExplainability] = useState(false);

  // Fetch series list
  const { data: seriesResponse, isLoading: seriesLoading } = useQuery({
    queryKey: ['dicomSeries', patientId],
    queryFn: () => dicomViewerAPI.listSeries({ patient_id: patientId }),
  });

  // Fetch metadata for selected series
  const { data: metadataResponse, isLoading: metadataLoading } = useQuery({
    queryKey: ['dicomMetadata', selectedSeries],
    queryFn: () => dicomViewerAPI.getSeriesMetadata(selectedSeries!),
    enabled: !!selectedSeries,
  });

  // Fetch current slice
  const { data: sliceResponse, isLoading: sliceLoading } = useQuery({
    queryKey: ['dicomSlice', selectedSeries, currentSlice, windowCenter, windowWidth],
    queryFn: () => dicomViewerAPI.getSlice(selectedSeries!, currentSlice, windowCenter, windowWidth),
    enabled: !!selectedSeries,
  });

  // Auto-select first series
  useEffect(() => {
    if (seriesResponse?.data?.series?.length > 0 && !selectedSeries) {
      setSelectedSeries(seriesResponse.data.series[0].series_id);
    }
  }, [seriesResponse, selectedSeries]);

  const handleSeriesSelect = (seriesId: string) => {
    setSelectedSeries(seriesId);
    setCurrentSlice(0); // Reset to first slice
  };

  const handleSliceChange = (delta: number) => {
    if (metadataResponse?.data) {
      const numSlices = metadataResponse.data.num_slices;
      const newSlice = Math.max(0, Math.min(numSlices - 1, currentSlice + delta));
      setCurrentSlice(newSlice);
    }
  };

  const renderSeriesList = () => {
    if (seriesLoading) return <div>Loading series...</div>;
    if (!seriesResponse?.data?.series) return <div>No series available</div>;

    return (
      <div className="series-list">
        <h3>Available Series ({seriesResponse.data.series.length})</h3>
        {seriesResponse.data.series.map((series: DicomSeries) => (
          <div
            key={series.series_id}
            className={`series-item ${selectedSeries === series.series_id ? 'selected' : ''}`}
            onClick={() => handleSeriesSelect(series.series_id)}
          >
            <div className="series-header">
              <span className="series-modality">{series.modality}</span>
              <span className="series-number">#{series.series_number}</span>
            </div>
            <div className="series-description">{series.series_description}</div>
            <div className="series-info">
              <span>{series.study_date}</span>
              <span>{series.num_instances} images</span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderMetadata = () => {
    if (!metadataResponse?.data) return null;

    const metadata = metadataResponse.data;

    return (
      <div className="metadata-panel">
        <h4>Series Metadata</h4>
        <div className="metadata-grid">
          <div className="metadata-item">
            <span className="label">Modality:</span>
            <span className="value">{metadata.modality}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Manufacturer:</span>
            <span className="value">{metadata.manufacturer}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Model:</span>
            <span className="value">{metadata.manufacturer_model_name}</span>
          </div>
          {metadata.magnetic_field_strength && (
            <div className="metadata-item">
              <span className="label">Field Strength:</span>
              <span className="value">{metadata.magnetic_field_strength}T</span>
            </div>
          )}
          {metadata.repetition_time && (
            <div className="metadata-item">
              <span className="label">TR:</span>
              <span className="value">{metadata.repetition_time}ms</span>
            </div>
          )}
          {metadata.echo_time && (
            <div className="metadata-item">
              <span className="label">TE:</span>
              <span className="value">{metadata.echo_time}ms</span>
            </div>
          )}
          <div className="metadata-item">
            <span className="label">Matrix:</span>
            <span className="value">{metadata.rows} × {metadata.columns}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Slices:</span>
            <span className="value">{metadata.num_slices}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Slice Thickness:</span>
            <span className="value">{metadata.slice_thickness}mm</span>
          </div>
        </div>
      </div>
    );
  };

  const renderViewer = () => {
    if (!selectedSeries) {
      return (
        <div className="viewer-placeholder">
          <p>Select a series to view</p>
        </div>
      );
    }

    const numSlices = metadataResponse?.data?.num_slices || 1;

    return (
      <div className="dicom-viewer-main">
        <div className="viewer-canvas">
          {sliceLoading ? (
            <div className="loading">Loading slice...</div>
          ) : (
            <div className="image-placeholder">
              <div className="image-info">
                <h3>DICOM Image Viewer</h3>
                <p>Image rendering would display here</p>
                <p>Libraries: Cornerstone.js, OHIF Viewer, or vtk.js</p>
                {sliceResponse && (
                  <div className="slice-info">
                    <p>Slice {currentSlice + 1} of {numSlices}</p>
                    {windowCenter && <p>Window: C={windowCenter} W={windowWidth}</p>}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="viewer-controls">
          <div className="slice-controls">
            <button onClick={() => handleSliceChange(-10)}>-10</button>
            <button onClick={() => handleSliceChange(-1)}>◀</button>
            <span className="slice-indicator">
              Slice {currentSlice + 1} / {numSlices}
            </span>
            <button onClick={() => handleSliceChange(1)}>▶</button>
            <button onClick={() => handleSliceChange(10)}>+10</button>
          </div>

          <div className="window-controls">
            <label>
              Window Center:
              <input
                type="number"
                value={windowCenter || 40}
                onChange={(e) => setWindowCenter(parseInt(e.target.value))}
                className="window-input"
              />
            </label>
            <label>
              Window Width:
              <input
                type="number"
                value={windowWidth || 400}
                onChange={(e) => setWindowWidth(parseInt(e.target.value))}
                className="window-input"
              />
            </label>
          </div>

          <div className="explainability-controls">
            <label>
              <input
                type="checkbox"
                checked={showExplainability}
                onChange={(e) => setShowExplainability(e.target.checked)}
              />
              <span>Show AI Explainability</span>
            </label>
          </div>
        </div>

        {renderMetadata()}
      </div>
    );
  };

  return (
    <div className="dicom-viewer">
      <header className="dicom-header">
        <h2>DICOM Viewer - Patient {patientId}</h2>
        <div className="dicom-actions">
          <button>Export</button>
          <button>Annotate</button>
          <button>Measure</button>
        </div>
      </header>

      <div className="dicom-content">
        <aside className="dicom-sidebar">
          {renderSeriesList()}
        </aside>

        <main className="dicom-main">
          {renderViewer()}
        </main>
      </div>

      <style>{`
        .dicom-viewer {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: #000;
          color: #fff;
        }

        .dicom-header {
          background: #1a1a1a;
          padding: 1rem 1.5rem;
          border-bottom: 1px solid #333;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .dicom-header h2 {
          margin: 0;
          font-size: 1.25rem;
          color: #fff;
        }

        .dicom-actions button {
          margin-left: 0.5rem;
          padding: 0.5rem 1rem;
          background: #333;
          color: #fff;
          border: 1px solid #444;
          border-radius: 4px;
          cursor: pointer;
        }

        .dicom-actions button:hover {
          background: #444;
        }

        .dicom-content {
          display: flex;
          flex: 1;
          overflow: hidden;
        }

        .dicom-sidebar {
          width: 300px;
          background: #1a1a1a;
          border-right: 1px solid #333;
          overflow-y: auto;
        }

        .dicom-main {
          flex: 1;
          display: flex;
          flex-direction: column;
        }

        .series-list {
          padding: 1rem;
        }

        .series-list h3 {
          margin-top: 0;
          margin-bottom: 1rem;
          font-size: 1rem;
          color: #aaa;
        }

        .series-item {
          background: #222;
          padding: 0.75rem;
          margin-bottom: 0.5rem;
          border-radius: 4px;
          border: 2px solid transparent;
          cursor: pointer;
          transition: all 0.2s;
        }

        .series-item:hover {
          background: #2a2a2a;
          border-color: #444;
        }

        .series-item.selected {
          border-color: #2196F3;
          background: #1a3a5a;
        }

        .series-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }

        .series-modality {
          background: #444;
          padding: 0.25rem 0.5rem;
          border-radius: 3px;
          font-size: 0.85rem;
        }

        .series-number {
          color: #888;
          font-size: 0.85rem;
        }

        .series-description {
          font-weight: 500;
          margin-bottom: 0.5rem;
        }

        .series-info {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
          color: #888;
        }

        .dicom-viewer-main {
          flex: 1;
          display: flex;
          flex-direction: column;
        }

        .viewer-canvas {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #000;
        }

        .viewer-placeholder {
          text-align: center;
          color: #666;
        }

        .image-placeholder {
          width: 512px;
          height: 512px;
          background: #111;
          border: 2px dashed #333;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .image-info {
          text-align: center;
          color: #666;
        }

        .slice-info {
          margin-top: 1rem;
          color: #888;
          font-size: 0.9rem;
        }

        .viewer-controls {
          background: #1a1a1a;
          border-top: 1px solid #333;
          padding: 1rem;
          display: flex;
          gap: 2rem;
          flex-wrap: wrap;
          align-items: center;
        }

        .slice-controls {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .slice-controls button {
          padding: 0.5rem 1rem;
          background: #333;
          color: #fff;
          border: 1px solid #444;
          border-radius: 4px;
          cursor: pointer;
          min-width: 40px;
        }

        .slice-controls button:hover {
          background: #444;
        }

        .slice-indicator {
          padding: 0 1rem;
          color: #aaa;
        }

        .window-controls {
          display: flex;
          gap: 1rem;
        }

        .window-controls label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #aaa;
          font-size: 0.9rem;
        }

        .window-input {
          width: 80px;
          padding: 0.25rem 0.5rem;
          background: #222;
          color: #fff;
          border: 1px solid #444;
          border-radius: 4px;
        }

        .explainability-controls label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
          color: #aaa;
        }

        .metadata-panel {
          background: #1a1a1a;
          border-top: 1px solid #333;
          padding: 1rem 1.5rem;
        }

        .metadata-panel h4 {
          margin-top: 0;
          margin-bottom: 1rem;
          color: #aaa;
          font-size: 0.9rem;
          text-transform: uppercase;
        }

        .metadata-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 0.75rem;
        }

        .metadata-item {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
        }

        .metadata-item .label {
          color: #888;
        }

        .metadata-item .value {
          color: #fff;
          font-weight: 500;
        }

        .loading {
          display: flex;
          align-items: center;
          justify-content: center;
          color: #666;
        }
      `}</style>
    </div>
  );
};

export default DicomViewer;
