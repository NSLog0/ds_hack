import React from 'react';
import { Link } from 'react-router-dom';
import { NotificationContainer } from 'react-notifications';

function AdminLayout(props) {
  return (
    <div>
      {props.children}
      <NotificationContainer />
    </div>
  );
}

export default AdminLayout;
