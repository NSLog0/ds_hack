import axios from 'axios';

let root = 'http://localhost:5000';

if (process.env.NODE_ENV === 'production') {
  root = 'http://localhost:8000';
}

export {};
