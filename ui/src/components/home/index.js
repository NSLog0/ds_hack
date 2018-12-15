import React from "react";
import { Link } from "react-router-dom";
import * as Cookies from "js-cookie";
import { get, isEmpty } from "lodash";
import moment from "moment";
// import { NotificationManager } from "react-notifications";

// import Modal from "../modals";
// import LineChart from "../charts/line";

// import {} from "../../helpers/http";

import {
  LOGIN_KEY,
  OFFICER_KEY,
  PATIEN_KEY,
  EXPIRES_TIME
} from "../../helpers/cookie";

class Home extends React.Component {
  state = {};

  componentDidMount = () => {};

  render() {
    return <div>Hello</div>;
  }
}

export default Home;
