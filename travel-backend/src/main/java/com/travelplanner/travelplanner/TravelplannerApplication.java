package com.travelplanner.travelplanner;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.annotation.Bean;

import javax.sql.DataSource;

@SpringBootApplication
public class TravelplannerApplication {

	public static void main(String[] args) {
		SpringApplication.run(TravelplannerApplication.class, args);
	}

	@Bean
	public DataSource dataSource() {
		return DataSourceBuilder.create()
				.url("jdbc:postgresql://localhost:5432/travelplanner")
				.username("postgres")
				.password("Asd12345")
				.driverClassName("org.postgresql.Driver")
				.build();
	}
}
