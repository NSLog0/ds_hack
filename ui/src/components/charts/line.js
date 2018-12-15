import React from 'react';
import * as d3 from 'd3';

class LineChart extends React.Component {

  state = {};

  componentDidMount() {
    let margin = { top: 5, right: 5, bottom: 5, left: 5 };
    let width = 250;
    let height = 50;

    let n = 10;

    let xScale = d3
      .scaleLinear()
      .domain([0, n - 1]) // input
      .range([0, width]); // output

    let yScale = d3
      .scaleLinear()
      .domain([0, 1]) // input
      .range([height, 0]); // output

    let line = d3
      .line()
      .x((d, i) => xScale(i)) // set the x values for the line generator
      .y(d => yScale(d.y)) // set the y values for the line generator
      .curve(d3.curveMonotoneX); // apply smoothing to the line

    let dataset = d3.range(n).map(() => ({ y: d3.randomUniform(1)() }));

    let svg = d3
      .select(this.refs.lineRef)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    svg
      .append('path')
      .datum(dataset) // 10. Binds data to the line
      .attr('class', 'line') // Assign a class for styling
      .attr('d', line); // 11. Calls the line generator
  }

  render() {
    return <div className="line-chart" ref="lineRef" />;
  }
}

export default LineChart;
