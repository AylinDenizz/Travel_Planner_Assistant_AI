package com.travelplanner.travelplanner.controller;

public class ReviewController {


    // Java Spring'de örnek REST çağrısı
    RestTemplate restTemplate = new RestTemplate();
    String flaskUrl = "http://localhost:5000/predict";
    Map<String, String> request = Map.of("text", "The hotel was dirty");

    HttpEntity<Map<String, String>> entity = new HttpEntity<>(request);
    ResponseEntity<Map> response = restTemplate.postForEntity(flaskUrl, entity, Map.class);
    Integer prediction = (Integer) response.getBody().get("prediction");

}
