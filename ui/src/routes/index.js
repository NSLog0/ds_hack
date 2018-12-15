import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';

import IndexPage from '../components/home/index';
import AdminLayout from '../layouts/admins';

const LayoutRoute = ({ component: Component, layout: Layout, ...rest }) => (
  <Route
    {...rest}
    render={props => (
      <Layout>
        <Component {...props} />
      </Layout>
    )}
  />
);

const Routers = () => (
  <Router>
    <Switch>
      <LayoutRoute exact path="/" layout={AdminLayout} component={IndexPage} />
    </Switch>
  </Router>
);

export default Routers;
