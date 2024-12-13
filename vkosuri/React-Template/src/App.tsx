import React from 'react';
import BarChart from './components/BarChart';
import PieChart from './components/PieChart';
import './style.css';

function App() {
  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>
          Insider Threat Dashboard - Analysis of User Behavioral Patterns and Risk Assessment
        </h1>
      </div>
      
      {/* Main Content */}
      <div className="dashboard-layout">
        <div className="bottom-section">
          {/* Bar Chart */}
          <div className="chart-container half-width">
            <div className="chart-wrapper bar-wrapper">
              <BarChart />
            </div>
          </div>
          
          {/* Pie Chart */}
          <div className="chart-container half-width">
            <div className="chart-wrapper pie-wrapper">
              <PieChart />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;