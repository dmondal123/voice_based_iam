import {data} from "../../logs_test/extracts.js"
import "./DataTable.css";

export default function DataTable()
{
    return (<>
    
    <div className="data-table">
    <h1>Impact</h1>
        <table className="table">
          <thead className="table__head">
            <tr className="table__heading">
              <th>USERID</th>
              <th>USER_NAME</th>
              <th>PASSWORD</th>
              <th>COOKIE</th>
            </tr>
          </thead>
          <tbody >
            {data.table.map((item) => (
              <tr key={item.USERID}>
                <td>{item.USERID}</td>
                <td>{item.USER_NAME}</td>
                <td>{item.PASSWORD}</td>
                <td>{item.COOKIE}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>{data.string1}</p>
        <p>{data.string2}</p>
        </div>
        </>
      );
    
}