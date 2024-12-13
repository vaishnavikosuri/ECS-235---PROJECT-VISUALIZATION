import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface InsiderThreatData {
  user_id: string;
  total_logons: number;
  risk_score: number;
  [key: string]: number | string;
}

const CHART_MARGINS = { top: 45, right: 180, bottom: 100, left: 80 };

const RISK_INDICATOR_KEYS = [
  'off_hours_ratio_normalized',
  'weekend_ratio_normalized',
  'hour_variance_normalized',
  'missing_disconnects_normalized',
  'suspicious_file_ops_normalized'
];

const LABEL_MAP: { [key: string]: string } = {
  'off_hours_ratio_normalized': 'Off Hours',
  'weekend_ratio_normalized': 'Weekend',
  'hour_variance_normalized': 'Hour Var',
  'missing_disconnects_normalized': 'Missing Disc',
  'suspicious_file_ops_normalized': 'Suspicious'
};

const BarChart: React.FC = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchAndRenderData = async () => {
      try {
        const response = await fetch('/data/insider_threat_analysis.csv');
        const csvData = await response.text();
        const insiderThreatData: InsiderThreatData[] = d3.csvParse(csvData, (d: any) => ({
          user_id: d.user_id,
          total_logons: +d.total_logons,
          risk_score: +d.risk_score,
          ...RISK_INDICATOR_KEYS.reduce((obj, key) => {
            obj[key] = typeof d[key] === 'string' ? +d[key] : d[key];
            return obj;
          }, {} as { [key: string]: number })
        }));

        const container = containerRef.current;
        if (!container) return;

        // Match the pie chart dimensions
        const width = 500;
        const height = 400;

        d3.select(svgRef.current).selectAll("*").remove();

        const svg = d3.select(svgRef.current)
          .attr('width', width)
          .attr('height', height)
          .attr('viewBox', `0 0 ${width} ${height}`)
          .attr('preserveAspectRatio', 'xMidYMid meet');

        const innerWidth = width - CHART_MARGINS.left - CHART_MARGINS.right;
        const innerHeight = height - CHART_MARGINS.top - CHART_MARGINS.bottom;

        const chart = svg.append('g')
          .attr('transform', `translate(${CHART_MARGINS.left}, ${CHART_MARGINS.top})`);

        const xScale = d3.scaleBand()
          .range([0, innerWidth])
          .padding(0.2);

        const yScale = d3.scaleLinear()
          .range([innerHeight, 0]);

        const userKeys = insiderThreatData.sort((a, b) => b.risk_score - a.risk_score).map(d => d.user_id);

        xScale.domain(RISK_INDICATOR_KEYS);

        const maxRiskIndicatorValues = RISK_INDICATOR_KEYS.map(key =>
          Math.max(...insiderThreatData.map(d => typeof d[key] === 'number' ? d[key] : 0))
        );

        const maxRiskIndicatorValue = Math.max(...maxRiskIndicatorValues);
        yScale.domain([0, 1.1]);

        chart.append('g')
          .attr('transform', `translate(0, ${innerHeight})`)
          .call(d3.axisBottom(xScale)
            .tickFormat(d => LABEL_MAP[d as string] || d as string))
          .selectAll('text')
          .style('fill', 'white')
          .style('font-size', '12px') // Match pie chart legend font size
          .style('font-weight', 'bold')
          .style('text-anchor', 'end')
          .attr('dx', '-.8em')
          .attr('dy', '.15em')
          .attr('transform', 'rotate(-45)');

        chart.append('g')
          .call(d3.axisLeft(yScale))
          .style('color', 'white')
          .selectAll('text')
          .style('fill', 'white')
          .style('font-size', '12px') // Match pie chart font size
          .style('font-weight', 'bold');

        chart.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('y', -60)
          .attr('x', -innerHeight / 2)
          .attr('text-anchor', 'middle')
          .style('fill', 'white')
          .style('font-size', '14px') // Adjusted for consistency
          .style('font-weight', 'bold')
          .text('Normalized Value');

        const colors = [
          '#FF0000',
          '#FFFF00',
          '#00FF00',
          '#00FFFF',
          '#0000FF'
        ];

        RISK_INDICATOR_KEYS.forEach((key, i) => {
          chart.selectAll(`.bar-${key}`)
            .data(userKeys)
            .enter()
            .append('rect')
            .attr('class', `bar-${key}`)
            .attr('x', d => xScale(key)!)
            .attr('width', xScale.bandwidth())
            .attr('y', d => {
              const value = insiderThreatData.find(user => user.user_id === d)?.[key];
              return typeof value === 'number' ? yScale(value) : 0;
            })
            .attr('height', d => {
              const value = insiderThreatData.find(user => user.user_id === d)?.[key];
              return typeof value === 'number' ? innerHeight - yScale(value) : 0;
            })
            .attr('fill', colors[i]);
        });

        const legend = chart.append('g')
          .attr('transform', `translate(${innerWidth + 10}, 0)`);

        RISK_INDICATOR_KEYS.forEach((key, i) => {
          const legendRow = legend.append('g')
            .attr('transform', `translate(0, ${i * 25})`); // Match pie chart legend spacing

          legendRow.append('rect')
            .attr('width', 15) // Match pie chart legend rect size
            .attr('height', 15)
            .attr('fill', colors[i]);

          legendRow.append('text')
            .attr('x', 24) // Match pie chart legend text position
            .attr('y', 12)
            .style('fill', 'white')
            .style('font-size', '12px') // Match pie chart legend font size
            .text(LABEL_MAP[key]);
        });

        // Match pie chart title style
        svg.append('text')
          .attr('x', width / 2)
          .attr('y', 30)
          .attr('text-anchor', 'middle')
          .style('fill', 'white')
          .style('font-size', '18px')
          .style('font-weight', 'bold')
          .text('Insider Threat Risk Indicators');

      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    fetchAndRenderData();

    const handleResize = () => {
      fetchAndRenderData();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div 
      ref={containerRef} 
      className="w-full h-full min-h-[400px] bg-black p-4 rounded-lg" // Match pie chart container styling
    >
      <svg
        ref={svgRef}
        className="w-full h-full"
      />
    </div>
  );
};

export default BarChart;