import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface InsiderThreatData {
  user_id: string;
  risk_level: string;
  total_logons: number;
  risk_score: number;
}

interface ChartData {
  riskLevel: string;
  percentage: number;
  count: number;
}

const RiskLevelPieChart = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const fetchAndRenderData = async () => {
      try {
        const response = await fetch('/data/insider_threat_analysis.csv');
        const csvData = await response.text();
        
        // Properly type the CSV parsing
        const rawData = d3.csvParse(csvData);
        const insiderThreatData: InsiderThreatData[] = rawData.map(row => ({
          user_id: row.user_id ?? '',
          risk_level: row.risk_level ?? '',
          total_logons: +(row.total_logons ?? 0),
          risk_score: +(row.risk_score ?? 0)
        }));

        // Calculate risk level distribution
        const riskLevelGroups = d3.group(insiderThreatData, d => d.risk_level.toLowerCase());
        const total = insiderThreatData.length;
        
        const data: ChartData[] = Array.from(riskLevelGroups, ([level, group]) => ({
          riskLevel: level.charAt(0).toUpperCase() + level.slice(1),
          percentage: (group.length / total) * 100,
          count: group.length
        }));

        const width = 500;
        const height = 400;
        const radius = Math.min(width, height) / 3;
        
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        svg
          .attr('width', width)
          .attr('height', height)
          .attr('viewBox', `0 0 ${width} ${height}`)
          .attr('preserveAspectRatio', 'xMidYMid meet');

        // Add title
        svg.append('text')
          .attr('x', width / 2)
          .attr('y', 30)
          .attr('text-anchor', 'middle')
          .style('fill', 'white')
          .style('font-size', '18px')
          .style('font-weight', 'bold')
          .text('Insider Threat Risk Level Distribution');

        // Color scale with proper typing - matching the image colors
        const colorScale = d3.scaleOrdinal<string, string>()
          .domain(['Low', 'Medium', 'High'])
          .range(['#32CD32', '#FFA500', '#FF0000']); // Adjusted colors to match image

        // Create pie layout with proper typing
        const pie = d3.pie<ChartData>()
          .value(d => d.percentage)
          .sort((a, b) => {
            const order = ['Low', 'Medium', 'High'];
            return order.indexOf(a.riskLevel) - order.indexOf(b.riskLevel);
          });

        // Create arc generator with proper typing
        const arc = d3.arc<d3.PieArcDatum<ChartData>>()
          .innerRadius(radius * 0.6)
          .outerRadius(radius);

        // Create chart group - centered in available space
        const g = svg.append('g')
          .attr('transform', `translate(${width/2}, ${height/2 + 20})`);

        // Add paths with proper typing
        g.selectAll('path')
          .data(pie(data))
          .enter()
          .append('path')
          .attr('d', arc)
          .attr('fill', d => colorScale(d.data.riskLevel))
          .attr('stroke', 'white')
          .attr('stroke-width', 1);

        // Add percentage labels
        g.selectAll('text.percentage')
          .data(pie(data))
          .enter()
          .append('text')
          .attr('class', 'percentage')
          .attr('transform', d => {
            const pos = arc.centroid(d);
            return `translate(${pos[0]}, ${pos[1]})`;
          })
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .style('fill', 'white')
          .style('font-size', '14px')
          .style('font-weight', 'bold')
          .text(d => `${d.data.percentage.toFixed(1)}%`);

        // Add legend
        const legend = svg.append('g')
          .attr('transform', `translate(${width - 80}, ${height/2 - 50})`);

        data.forEach((item, i) => {
          const legendRow = legend.append('g')
            .attr('transform', `translate(0, ${i * 25})`);

          legendRow.append('rect')
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', colorScale(item.riskLevel));

          legendRow.append('text')
            .attr('x', 24)
            .attr('y', 12)
            .style('fill', 'white')
            .style('font-size', '12px')
            .text(item.riskLevel);
        });

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
    <div className="w-full h-full min-h-[400px] bg-black p-4 rounded-lg">
      <svg 
        ref={svgRef}
        className="w-full h-full"
      />
    </div>
  );
};

export default RiskLevelPieChart;