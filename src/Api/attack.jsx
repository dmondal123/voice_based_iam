import axios from 'axios';


const basePath = 'https://jsonplaceholder.typicode.com'
export async function getThreat() {
    try {
      const response = await axios.get(`${basePath}/posts/1`);
      console.log(response);
      
      return response;

    } catch (error) {
      console.error(error);
    }
  }
  