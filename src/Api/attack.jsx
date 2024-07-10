// ONLY FOR FRONTEND

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


  //CODE AFTER INTEGRATION WITH BACKEND
//   import axios from 'axios';
 
 
// const basePath = 'http://localhost:8080/api/send-config'
// export async function getThreat() {
 
 
//   const configObject = {
//     appName: 'webgoat',
//     attackType: 'sql injection'
//   };
//     try {
//       const response = await axios.post(basePath,configObject);
//       console.log('Response from backend:', response.data);
//       return response.data;
//     }
 
//     catch (error) {
//       console.error('Error sending config to backend:', error);
//       throw error; // Optional: rethrow the error or handle it as needed
//     }
//   }
  