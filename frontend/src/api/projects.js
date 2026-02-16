import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const getHeaders = (token) => ({
    Authorization: `Bearer ${token}`
});

export const projectsApi = {
    // Get all projects
    getAll: async (token) => {
        const response = await axios.get(`${API}/projects`, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Get single project
    getOne: async (token, projectId) => {
        const response = await axios.get(`${API}/projects/${projectId}`, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Create project
    create: async (token, data) => {
        const response = await axios.post(`${API}/projects`, data, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Update project
    update: async (token, projectId, data) => {
        const response = await axios.patch(`${API}/projects/${projectId}`, data, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Delete project
    delete: async (token, projectId) => {
        const response = await axios.delete(`${API}/projects/${projectId}`, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Upload audio
    uploadAudio: async (token, projectId, audioBlob) => {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        
        const response = await axios.post(`${API}/projects/${projectId}/upload`, formData, {
            headers: {
                ...getHeaders(token),
                'Content-Type': 'multipart/form-data'
            }
        });
        return response.data;
    },

    // Analyze audio
    analyze: async (token, projectId) => {
        const response = await axios.post(`${API}/projects/${projectId}/analyze`, {}, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Generate beat
    generateBeat: async (token, projectId) => {
        const response = await axios.post(`${API}/projects/${projectId}/generate`, {}, {
            headers: getHeaders(token)
        });
        return response.data;
    },

    // Check beat status
    checkBeatStatus: async (token, projectId) => {
        const response = await axios.get(`${API}/projects/${projectId}/beat-status`, {
            headers: getHeaders(token)
        });
        return response.data;
    }
};

export const genresApi = {
    getAll: async () => {
        const response = await axios.get(`${API}/genres`);
        return response.data.genres;
    }
};
