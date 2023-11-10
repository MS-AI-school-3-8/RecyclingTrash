//이미지 도출 소스 코드
function setThumbnail(event) {
    var reader = new FileReader();

    reader.onload = function(event) {
      var img = document.createElement("img");
      img.setAttribute("src", event.target.result);
      document.querySelector("div#image_container").appendChild(img);
    };

    reader.readAsDataURL(event.target.files[0]);
}



var submit = document.getElementById('submitButton');
submit.onclick = showImage;     //Submit 버튼 클릭시 이미지 보여주기

/*
function getImage() {
    var newImage = document.getElementById('image-preview').lastElementChild;    
}
*/



const getImage = async () => {
    const container = document.getElementById("img-container")
    
    const formData = new FormData();
    formData.append('file', event.target.files[0]);

    const response  = await axios.post("http://172.30.1.89:5000/object-detection", formData);

    console.log(response);
    const blobImg = await response.blob();
    console.log(blobImg);
    const imgUrl = URL.createObjectURL(blobImg);
    console.log(imgUrl);
    const html = `<img src="${imgUrl}" alt="">`;
    container.innerHTML = html;
    
}
window.onload = () => { getImage(); };


