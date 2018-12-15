import React from 'react';
import ReactDOM from 'react-dom';
import { unregister as unregisterServiceWorker } from './registerServiceWorker'
import 'react-notifications/lib/notifications.css';

import './index.scss';

import Router from './routes/index';

ReactDOM.render(<Router />, document.getElementById('root'));
unregisterServiceWorker();
