export default { server: { port: 5173, proxy: { "/api": "http://localhost:8000", "/results": "http://localhost:8000" } } };
